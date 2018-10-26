import pandas as pd
import numpy as np
import os.path
from my_classes import DataGenerator
import argparse

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.losses import mse, binary_crossentropy


def preprocess_df(_path_to_csv, path_to_data, colour_band, file_extension):
    _df = pd.read_csv(_path_to_csv)
    print('successfully loaded file: ', _path_to_csv)

    # fill NaN as 0
    _df = _df.fillna(0)

    _df = _df.rename(index=str, columns={
        'vuosi': 'YEAR',
        'lohkonro': 'field parcel',
        'tunnus': 'identifier',
        'kasvikoodi': 'PLANT CODE',
        'kasvi': 'PLANT',
        'lajikekood': 'VARIETY CODE',
        'lajike': 'VARIETY',
        'pintaala': 'Property area',
        'tays_tuho': 'full crop loss',
        'ositt_tuho': 'partial crop loss'})

    # remove duplicated entries in field parcel
    # print(_df.shape[0] - len(np.unique(_df['field parcel'])), 'duplicate entries')
    fieldparcel = _df['field parcel']
    _df = _df[fieldparcel.duplicated() == False]
    # print(_df.shape[0] - len(np.unique(_df['field parcel'])), 'duplicate entries')

    # print('total number of fields: ', _df.shape[0])

    # print('create new column: relative crop loss = crop loss / Property area')
    _df['full crop loss scaled'] = _df['full crop loss'] / _df['Property area']
    _df['partial crop loss scaled'] = _df['partial crop loss'] / _df['Property area']

    # select largest number of samples for one given plant species
    plants = _df['PLANT']
    num = 0
    for plant in np.unique(list(plants)):
        num_tmp = len(_df[plants == plant])
        # print(plant, '\t ', num_tmp)

        if num_tmp > num:
            num = num_tmp
            plant_max = plant
    # print('maximum number for', plant_max, 'with', num, 'entries')
    _df = _df[plants == plant_max]

    col_list = ['field parcel', 'full crop loss scaled', 'partial crop loss scaled']

    # print('trim data frame to:', col_list)
    _df = _df[col_list]

    print('total number of fields before verifying file existence ', _df.shape[0])
    # check if files for fields exist, if not, remove from data frame
    not_existing = []
    for index, row in _df.iterrows():
        file = path_to_data + row['field parcel'] + '_' + colour_band + file_extension
        if not os.path.isfile(file):
            not_existing.append(index)

    _df = _df.drop(not_existing)

    print('data frame created with a total number of fields: ', _df.shape[0])
    return _df


def split_dataframe(_df, train_p, val_p, random_state=200):
    # split data frame into train, validation and test
    _train_df = _df.sample(frac=train_p, random_state=random_state)
    _df = _df.drop(_train_df.index)
    _val_df = _df.sample(frac=val_p / (1 - train_p), random_state=random_state)
    _test_df = _df.drop(_val_df.index)

    # make field parcel as indeces
    _train_df = _train_df.set_index('field parcel')
    _val_df = _val_df.set_index('field parcel')
    _test_df = _test_df.set_index('field parcel')

    return _train_df, _val_df, _test_df


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(_args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    _z_mean, _z_log_var = _args
    batch = K.shape(_z_mean)[0]
    dim = K.int_shape(_z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return _z_mean + K.exp(0.5 * _z_log_var) * epsilon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()

    # Parameters
    params = {
        'path_to_csv': 'MAVI2/2015/rap_2015.csv',
        'train_p': 0.8,
        'val_p': 0.1,
        'path_to_data': 'data/',
        'colour_band': 'RAW',
        'file_extension': '.tiff',
        'dim': (512, 512),
        'batch_size': 16,
        'n_channels': 13,
        'shuffle': True,
        'kernel_size': 3,
        'filters': 16,
        'latent_dim': 2,
        'epochs': 30
    }

    df = preprocess_df(params['path_to_csv'], params['path_to_data'], params['colour_band'], params['file_extension'])

    # add color spectrum and file extension to field parcel column in order to match to the file name
    # df['field parcel'] = df['field parcel'] + '_' + color_spectrum

    # split data frame into train, validation and test
    train_df, val_df, test_df = split_dataframe(df, params['train_p'], params['val_p'], )
    del df

    # create dictionaries of IDs and labels
    partition = {
        'train': train_df.index.tolist(), 'validation': val_df.index.tolist(), 'test': test_df.index.tolist()
    }
    labels = {**train_df['full crop loss scaled'].to_dict(), **val_df['full crop loss scaled'].to_dict(),
              **test_df['full crop loss scaled'].to_dict()}

    # Generators
    print('create DataGenerators')
    training_generator = DataGenerator(partition['train'], labels, params['path_to_data'], params['colour_band'],
                                       params['file_extension'], params['batch_size'], params['dim'],
                                       params['n_channels'], params['shuffle'])
    validation_generator = DataGenerator(partition['validation'], labels, params['path_to_data'], params['colour_band'],
                                         params['file_extension'], params['batch_size'], params['dim'],
                                         params['n_channels'], params['shuffle'])

    # network parameters
    input_shape = (params['dim'][0], params['dim'][1], params['n_channels'])
    kernel_size = params['kernel_size']
    filters = params['filters']
    latent_dim = params['latent_dim']
    epochs = params['epochs']

    print('create graph / model')
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(2):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(2):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=params['n_channels'],
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')


    def my_vae_loss(_inputs, _outputs):
        # VAE loss = mse_loss or xent_loss + kl_loss
        if args.mse:
            reconstruction_loss = mse(K.flatten(_inputs), K.flatten(_outputs))
        else:
            reconstruction_loss = binary_crossentropy(K.flatten(_inputs),
                                                      K.flatten(_outputs))

        reconstruction_loss *= params['dim'][0] * params['dim'][1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss


    # vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop', loss=my_vae_loss)
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            epochs=epochs,
            use_multiprocessing=True,
            workers=6)
        vae.save_weights('vae_cnn_mnist.h5')
