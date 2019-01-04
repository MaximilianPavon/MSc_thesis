import pandas as pd
import numpy as np
import os
from keras import backend as K
from tqdm import tqdm
import matplotlib.pyplot as plt


def preprocess_df(path_to_csv, path_to_data, colour_band, file_extension):
    df = pd.read_csv(path_to_csv)
    print('successfully loaded file: ', path_to_csv)

    # fill NaN as 0
    df = df.fillna(0)

    df = df.rename(index=str, columns={
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
    # print(df.shape[0] - len(np.unique(df['field parcel'])), 'duplicate entries')
    fieldparcel = df['field parcel']
    df = df[fieldparcel.duplicated() == False]
    # print(df.shape[0] - len(np.unique(df['field parcel'])), 'duplicate entries')

    # print('total number of fields: ', df.shape[0])

    # print('create new column: relative crop loss = crop loss / Property area')
    df['full crop loss scaled'] = df['full crop loss'] / df['Property area']
    df['partial crop loss scaled'] = df['partial crop loss'] / df['Property area']

    # select largest number of samples for one given plant species
    plants = df['PLANT']
    num = 0
    for plant in np.unique(list(plants)):
        num_tmp = len(df[plants == plant])
        # print(plant, '\t ', num_tmp)

        if num_tmp > num:
            num = num_tmp
            plant_max = plant
    # print('maximum number for', plant_max, 'with', num, 'entries')
    df = df[plants == plant_max]

    col_list = ['field parcel', 'full crop loss scaled', 'partial crop loss scaled']

    # print('trim data frame to:', col_list)
    df = df[col_list]

    # check if files for fields exist, if not, remove from data frame
    # write relative path including colour band and file extension to dataframe for future usage
    print('total number of fields before verifying file existence ', df.shape[0])
    subfolders = ['dataset1/', 'dataset2/', 'dataset3/', 'dataset4/', 'dataset5/', 'dataset6/', 'dataset7/', 'dataset8/']
    not_existing = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        for folder in subfolders:
            file_full = os.path.join(path_to_data, folder) + row['field parcel'] + '_' + colour_band + file_extension
            file_partial = folder + row['field parcel'] + '_' + colour_band + file_extension
            if os.path.isfile(file_full):
                # df.at[index, 'full path'] = file_full
                df.at[index, 'partial path'] = file_partial
                break
            elif folder == subfolders[-1]:
                not_existing.append(index)

    df = df.drop(not_existing)

    print('data frame created with a total number of fields: ', df.shape[0])
    return df


def split_dataframe(df, train_p, val_p, random_state=200):
    # split data frame into train, validation and test
    train_df = df.sample(frac=train_p, random_state=random_state)
    df = df.drop(train_df.index)
    val_df = df.sample(frac=val_p / (1 - train_p), random_state=random_state)
    test_df = df.drop(val_df.index)

    # make field parcel as indeces including the relative path -> partial path
    train_df = train_df.set_index('partial path')
    val_df = val_df.set_index('partial path')
    test_df = test_df.set_index('partial path')

    return train_df, val_df, test_df


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    reparameterization trick
    instead of sampling from Q(z|X), sample eps = N(0,I)
    then z = z_mean + sqrt(var)*eps

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(decoder, data_generator=None, batch_size=128, model_name='../4_runs/plots/vae_latent'):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        decoder : decoder model
        data_generator (tuple): test data_generator
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    os.makedirs(model_name, exist_ok=True)

    # print('display a 2D plot of the digit classes in the latent space')
    # filename = os.path.join(model_name, "vae_mean.png")
    # z_mean, _, _ = encoder.predict_generator(data_generator, batch_size=batch_size)
    # plt.figure(figsize=(12, 10))
    # plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    # plt.colorbar()
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.savefig(filename)
    # plt.show()

    print('display a 30x30 2D manifold of reconstructed satellite images')
    filename = os.path.join(model_name, "digits_over_latent.png")
    n = 30
    img_size = 512
    n_channels = 13
    figure = np.zeros((img_size * n, img_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            image = x_decoded[0].reshape(img_size, img_size, n_channels)
            image = image[:, :, [4, 3, 2]]
            figure[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size, :] = image

    plt.figure(figsize=(10, 10))
    start_range = img_size // 2
    end_range = n * img_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, img_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    plt.savefig(filename)
    plt.show()
