from my_functions import get_available_gpus, sampling, plot_latent_space, create_dataset, get_device_util, get_device_id, get_OS
from my_classes import MyCallbackDecoder, MyCallbackCompOrigDecoded
import tensorflow as tf
import datetime
import argparse
import sys
import os

op_sys = get_OS()
if op_sys == 'Darwin':
    # fix for macOS issue regarding library import error:
    # OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group(title='required arguments')
    req_grp.add_argument("-c", "--computer", help="Specify computer: use \'triton\', \'mac\' or \'workstation\'.",
                         required=True)
    parser.add_argument("-p", "--project_path", help="Specify project path, where the project is located.")
    parser.add_argument("-d", "--data_path",
                        help="Specify path, where the data is located. E.g. /tmp/$SLURM_JOB_ID/05_images_masked/ ")
    parser.add_argument("-m", "--model", help="Load compiled models (.hdf5 file) saved by model.save(filepath). Path "
                                              "until parent directory: e.g \'4_runs/logging/models/")
    parser.add_argument("-w", "--weights", help="Load trained weights (.h5 file) saved by model.save_weights(filepath)."
                                                "Path until parent directory: e.g \'4_runs/logging/weights/")
    parser.add_argument("--mse", action='store_true', help="Use mse loss instead of binary cross entropy (default)")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode - reduces epochs and steps_per_epoch")
    args = parser.parse_args()

    if not args.project_path:
        if args.computer == 'triton':
            args.project_path = '/scratch/cs/ai_croppro'
        elif args.computer == 'mac':
            args.project_path = '/Users/maximilianproll/Dropbox (Aalto)/'
        elif args.computer == 'workstation':
            args.project_path = '/m/cs/scratch/ai_croppro'
        else:
            sys.exit('Please specify the computer this programme runs on using \'triton\', \'mac\' or \'workstation\'')

    if args.model and not args.project_path in args.model:

        if '..' in args.model:
            # remove leading '..'
            args.model = '/'.join(args.model.split('/')[1:])

        args.model = os.path.join(args.project_path, args.model)

    if args.weights and not args.project_path in args.weights:

        if '..' in args.weights:
            # remove leading '..'
            args.weights = '/'.join(args.weights.split('/')[1:])

        args.weights = os.path.join(args.project_path, args.weights)

    print(f'available GPUs:) ')
    gpu_pci_bus_id = get_available_gpus()
    gpu_device_ID = get_device_id(gpu_pci_bus_id)

    # Parameters
    params = {
        'path_to_csv': os.path.join(args.project_path,
                                    '2_data/01_MAVI_unzipped_preprocessed/MAVI2/2015/preprocessed_masked.csv'),
        'train_p': 0.8,
        'val_p': 0.1,
        'path_to_data': args.data_path if args.data_path else os.path.join(args.project_path,
                                                                           '2_data/05_images_masked/'),
        'has_cb_and_ext': True,
        'colour_band': 'BANDS-S2-L1C',
        'file_extension': '.tiff',
        'dim': (512, 512),
        'batch_size': 16,
        'n_channels': 13,
        'shuffle': True,
        'n_Conv': 6,
        'kernel_size': 3,
        'filters': 20,
        'latent_dim': 4,
        'epochs': 70
    }
    n_parallel_readers = 4
    ds_train, steps_per_epoch_train = create_dataset(params['path_to_data'], 'train', params['batch_size'],
                                                     params['batch_size'], n_parallel_readers )
    ds_val, steps_per_epoch_val = create_dataset(params['path_to_data'], 'val', params['batch_size'],
                                                 params['batch_size'], n_parallel_readers)
    ds_test, steps_per_epoch_test = create_dataset(params['path_to_data'], 'test', params['batch_size'],
                                                   params['batch_size'], n_parallel_readers)

    # network parameters
    input_shape = (params['dim'][0], params['dim'][1], params['n_channels'])
    kernel_size = params['kernel_size']
    filters = params['filters']
    latent_dim = params['latent_dim']
    epochs = params['epochs']

    print('create graph / model')
    # VAE model = encoder + decoder
    # build encoder model
    inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(params['n_Conv']):
        # filters *= 2
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=kernel_size,
                                   activation='relu',
                                   strides=2,
                                   padding='same')(x)
        # x = tf.keras.layers.BatchNormalization()(x)

    # shape info needed to build decoder model
    shape = tf.keras.backend.int_shape(x)

    # generate latent vector Q(z|X)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    tf.keras.utils.plot_model(encoder, to_file=os.path.join(args.project_path, '4_runs/plots/vae_encoder.png'),
                              show_shapes=True)

    # build decoder model
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    x = tf.keras.layers.Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(params['n_Conv']):
        # filters //= 2
        x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                            kernel_size=kernel_size,
                                            activation='relu',
                                            strides=2,
                                            padding='same')(x)
        # x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Conv2DTranspose(filters=params['n_channels'],
                                              kernel_size=kernel_size,
                                              activation='sigmoid',
                                              padding='same',
                                              name='decoder_output')(x)

    # instantiate decoder model
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    tf.keras.utils.plot_model(decoder, to_file=os.path.join(args.project_path, '4_runs/plots/vae_decoder.png'),
                              show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = tf.keras.models.Model(inputs, outputs, name='vae')


    def my_vae_loss(_inputs, _outputs):
        # VAE loss = mse_loss or xent_loss + kl_loss
        if args.mse:
            reconstruction_loss = tf.keras.losses.mse(
                tf.keras.backend.flatten(_inputs), tf.keras.backend.flatten(_outputs))
        else:
            reconstruction_loss = tf.keras.losses.binary_crossentropy(
                tf.keras.backend.flatten(_inputs), tf.keras.backend.flatten(_outputs))

        reconstruction_loss *= params['dim'][0] * params['dim'][1]
        kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
        return vae_loss

    rmsprop = tf.keras.optimizers.RMSprop(lr=0.00001)
    vae.compile(optimizer=rmsprop, loss=my_vae_loss, metrics=['accuracy'])
    vae.summary()
    tf.keras.utils.plot_model(vae, to_file=os.path.join(args.project_path, '4_runs/plots/vae.png'), show_shapes=True)

    # folder extension for bookkeeping
    datetime_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_string = str(params['n_Conv']) + 'n_Conv_' + datetime_string

    # add callbacks for:
    # - creating saving decoded image after log_freq epochs
    # - TensorBoard logger
    # - logging and for saving model checkpoints

    mycb_comparison = MyCallbackCompOrigDecoded(
        log_dir=os.path.join(args.project_path, '4_runs/plots/comparison_' + config_string),
        dataset=ds_test, num_examples=10,  log_freq=1)

    mycb_decoder = MyCallbackDecoder(
        encoder, decoder,
        log_dir=os.path.join(args.project_path, '4_runs/plots/decoder_' + config_string),
        num_examples_to_generate=16, log_freq=1
    )

    tbCallBack = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.project_path, '4_runs/logging/TBlogs/' + config_string),
        histogram_freq=0,  # TODO: fix error when setting histogram_freq > 0
        batch_size=params['batch_size'],
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.project_path, '4_runs/logging/checkpoints/vae_' + config_string + '.hdf5'),
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1,
    )

    callbacks_list = [mycb_comparison, mycb_decoder, model_checkpoint, tbCallBack]

    if args.weights:
        print(f'loading weights from: {args.weights}')
        vae = vae.load_weights(os.path.join(args.weights, 'vae.h5'))
        encoder = encoder.load_weights(os.path.join(args.weights, 'encoder.h5'))
        decoder = decoder.load_weights(os.path.join(args.weights, 'decoder.h5'))

    elif args.model:
        print(f'loading models from: {args.model}')
        vae = tf.keras.models.load_model(os.path.join(args.model, 'vae.hdf5'),
                                         custom_objects={'my_vae_loss': my_vae_loss})
        encoder = tf.keras.models.load_model(os.path.join(args.model, 'encoder.hdf5'))
        decoder = tf.keras.models.load_model(os.path.join(args.model, 'decoder.hdf5'))
        print('models loaded')
    else:
        # train the autoencoder
        print('start the training')
        vae.fit(
            ds_train,
            steps_per_epoch=steps_per_epoch_train if not args.debug else 3,
            validation_data=ds_val,
            validation_steps=steps_per_epoch_val if not args.debug else 3,
            epochs=epochs if not args.debug else 3,
            callbacks=callbacks_list,
        )
        print('training done')

        # save models and weights
        dir_models = os.path.join(args.project_path, '4_runs/logging/models/', config_string)
        os.makedirs(dir_models, exist_ok=True)
        vae.save(os.path.join(dir_models, 'vae.hdf5'))
        encoder.save(os.path.join(dir_models, 'encoder.hdf5'))
        decoder.save(os.path.join(dir_models, 'decoder.hdf5'))

        dir_weights = os.path.join(args.project_path, '4_runs/logging/weights/', config_string)
        os.makedirs(dir_weights, exist_ok=True)
        vae.save_weights(os.path.join(dir_weights, 'vae.h5'))
        encoder.save_weights(os.path.join(dir_weights, 'encoder.h5'))
        decoder.save_weights(os.path.join(dir_weights, 'decoder.h5'))
        print('models and weights saved')

    # define example images and their information tag for plotting them in the latent space
    example_images = [
        os.path.join(args.project_path, '2_data/05_images_masked/dataset1/0040491234-A_BANDS-S2-L1C.tiff'),
        os.path.join(args.project_path, '2_data/05_images_masked/dataset1/8930312979-A_BANDS-S2-L1C.tiff'),
        os.path.join(args.project_path, '2_data/05_images_masked/dataset1/0090248594-A_BANDS-S2-L1C.tiff'),
        os.path.join(args.project_path, '2_data/05_images_masked/dataset1/9810286471-A_BANDS-S2-L1C.tiff')
    ]

    if op_sys == 'Darwin':
        example_images = [
            os.path.join(args.project_path, '2_data/05_images_masked/dataset8/0050496277-A_BANDS-S2-L1C.tiff'),
            os.path.join(args.project_path, '2_data/05_images_masked/dataset8/0050496277-A_BANDS-S2-L1C.tiff'),
            os.path.join(args.project_path, '2_data/05_images_masked/dataset8/0050496277-A_BANDS-S2-L1C.tiff'),
            os.path.join(args.project_path, '2_data/05_images_masked/dataset8/0050496277-A_BANDS-S2-L1C.tiff')
        ]

    ex_im_informations = [
        'only full crop loss',
        'both partial and full crop loss',
        'no loss',
        'only partial crop loss',
    ]
    plot_latent_space((encoder, decoder),
                      dataset=ds_test,
                      example_images=example_images,
                      ex_im_informations=ex_im_informations,
                      path=os.path.join(args.project_path, '4_runs/plots/latent_' + config_string)
                      )
