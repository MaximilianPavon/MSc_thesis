import pandas as pd
import numpy as np
import os
from keras import backend as K
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from skimage import io, exposure


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
    subfolders = ['dataset1/', 'dataset2/', 'dataset3/', 'dataset4/', 'dataset5/', 'dataset6/', 'dataset7/',
                  'dataset8/']
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


def plot_latent_space(model, data_generator, example_images, ex_im_informations,  path='../4_runs/plots/latent/'):
    """Plots labels and satellite images as function of 2-dim latent vector

    # Arguments:
        :param model: tuple of encoder and decoder model
        :param data_generator: test data_generator
        :param example_images: path to the example images to display in the 2D latent representation
        :param ex_im_informations: information for the example_images to which class they belong
        :param path: path for saving the plots
    """

    encoder, decoder = model

    img_size = 512
    n_channels = 13

    os.makedirs(path, exist_ok=True)

    print('display a 2D plot of the satellite images  projected into the latent space')
    filename = os.path.join(path, "z_mean_over_latent.png")
    z_mean, _, _ = encoder.predict_generator(data_generator, verbose=1, steps=data_generator.step_size)

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(z_mean[:, 0], z_mean[:, 1], s=0.2, zorder=1)

    x_min = np.min(z_mean, axis=0)[0]
    x_max = np.max(z_mean, axis=0)[0]
    y_min = np.min(z_mean, axis=0)[1]
    y_max = np.max(z_mean, axis=0)[1]

    plot_margin_x = (x_max - x_min) * 0.1  # add 2 x 10% to the x range for displaying the images
    plot_margin_y = (y_max - y_min) * 0.1  # add 2 x 10% to the y range for displaying the images

    ax.set_xlim(left=x_min - plot_margin_x, right=x_max + plot_margin_x)
    ax.set_ylim(bottom=y_min - plot_margin_y, top=y_max + plot_margin_y)
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")

    for file, info in zip(example_images, ex_im_informations):
        im = io.imread(file)

        z_mean, _, _ = encoder.predict(
            np.reshape(im, (1, img_size, img_size, n_channels))
            # reshape necessary because the encoder expects a 4D array with batch x img_size x img_size x n_channels
        )
        z_mean = z_mean[0]

        ax.scatter(z_mean[0], z_mean[1], s=10)

        # display information tag of the image
        textbox = TextArea(info, minimumdescent=False)

        ab = AnnotationBbox(textbox, z_mean,
                            xybox=(-50, 90),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            )
        ax.add_artist(ab)

        # display the image itself
        im_rgb = im[:, :, [3, 2, 1]]
        im_rgb = exposure.rescale_intensity(im_rgb)
        imagebox = OffsetImage(im_rgb, zoom=0.1)

        ab = AnnotationBbox(imagebox, z_mean,
                            xybox=(-50., 50.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            arrowprops=dict(arrowstyle="->"))

        ax.add_artist(ab)

    plt.title('Satellite images projected into the latent space')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

    n = 30
    print(f'display a {n}x{n} 2D manifold of reconstructed satellite images')
    filename = os.path.join(path, "images_over_latent.png")
    figure = np.zeros((img_size * n, img_size * n, 3))
    # linearly spaced coordinates corresponding to the 2D plot
    # of satellite images in the latent space
    grid_x = np.linspace(x_min - plot_margin_x, x_max + plot_margin_x, n)
    grid_y = np.linspace(y_min - plot_margin_y, y_max + plot_margin_y, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            im = x_decoded[0].reshape(img_size, img_size, n_channels)
            im = im[:, :, [3, 2, 1]]
            im = exposure.rescale_intensity(im)

            # a tiff file contains the raw channels: return [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12]
            # a rgb images needs the following channels:  return [B04, B03, B02]
            # ==> index [3, 2, 1]

            figure[i * img_size: (i + 1) * img_size, j * img_size: (j + 1) * img_size, :] = im

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
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
