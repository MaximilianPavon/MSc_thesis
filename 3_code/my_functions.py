import io
import os
import platform
import sys
from subprocess import Popen, PIPE

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from skimage import exposure
from skimage import io as ski_io
from sklearn.metrics import confusion_matrix
from tensorflow.python.client import device_lib
from tqdm import tqdm


def get_OS():
    return platform.system()


def get_available_gpus():
    # if get_OS() == 'Darwin':
    #     return ''
    #
    # stdout = sys.stdout
    # sys.stdout = io.StringIO()
    #
    # local_device_protos = device_lib.list_local_devices()
    # print(local_device_protos)  # printing is necessary to pass information to stdout
    #
    # # get output and restore sys.stdout
    # tf_gpu_info = sys.stdout.getvalue()
    # sys.stdout = stdout
    #
    # tf_gpu_info = tf_gpu_info.split(os.linesep)[-3]
    # # tf_gpu_info is: 'physical_device_desc: "device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:18:00.0, compute capability: 7.0"'
    #
    # tf_gpu_info = tf_gpu_info.split('pci bus id:')[-1]
    # # tf_gpu_info is: ' 0000:18:00.0, compute capability: 7.0"'
    #
    # tf_gpu_pci_bus_id = tf_gpu_info.split(', compute capability')[0].strip()
    # print(f'pci bus id: {tf_gpu_pci_bus_id}')

    local_device_protos = device_lib.list_local_devices()
    print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    return
    # return tf_gpu_pci_bus_id


def _get_serials_pci_bus_ids():
    p = Popen(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,serial", "--format=csv,noheader"], stdout=PIPE)

    stdout, stderror = p.communicate()

    output = stdout.decode('UTF-8')
    # output looks like this:
    # index, uuid, pci.bus_id, serial
    # 0, GPU-f186d30d-edbf-6153-0d12-0c9e968f25ea, 00000000:18:00.0, 0423718059721
    # 1, GPU-0964a27f-d918-b7b5-7cdb-f557ab54d74e, 00000000:3B:00.0, 0423718082092
    # 2, GPU-7eb57620-43df-977b-0f8d-2b90efa4f2f1, 00000000:86:00.0, 0423718059629
    # 3, GPU-5f65db4d-3355-24bf-0888-ec896a6dbceb, 00000000:AF:00.0, 0423718081271

    lines = output.split(os.linesep)
    # print(lines)
    numDevices = len(lines) - 1

    pci_bus_ids, serials = [], []

    for g in range(numDevices):
        line = lines[g]
        vals = line.split(', ')
        for i in range(len(vals)):
            if (i == 2):
                pci_bus_id = vals[i]
            elif (i == 3):
                serial = vals[i]
        pci_bus_ids.append(pci_bus_id)
        serials.append(serial)

    return serials, pci_bus_ids


def _get_gpu_util_str():
    # redirect sys.stdout to a buffer
    stdout = sys.stdout
    sys.stdout = io.StringIO()

    GPUtil.showUtilization()

    # get output and restore sys.stdout
    gpu_util_str = sys.stdout.getvalue()
    sys.stdout = stdout
    # returns something like this" ['| ID | GPU | MEM |', '------------------', '|  0 |  0% |  2% |', '']
    #                                   header          , header separation   , values of GPU(s)
    return gpu_util_str.split('\n')


def get_device_id(gpu_pci_bus_id):
    if get_OS() == 'Darwin':
        return 0

    serials, pci_bus_ids = _get_serials_pci_bus_ids()
    for i, pci in enumerate(pci_bus_ids):
        if gpu_pci_bus_id in pci:
            id = i
            break

    return id


def get_device_util(deviceID):
    if platform.system() == 'Darwin':
        #  check if run on macOS with no available GPU and exit the function returning constant -100 utilisation
        #  as calling GPUtil.showUtilization() with no availble GPU will throw an error
        return -100

    # offset by two lines due to header and header separation
    device_str = _get_gpu_util_str()[deviceID + 2]
    # device_str is something like: '|  0 |  0% |  2% |'

    _, id_nr, gpu_util, mem_util, _ = device_str.split('|')
    # gpu_util is something like: '  0% '

    gpu_util = gpu_util.strip()
    # gpu_util is something like: '0%'

    gpu_util = int(gpu_util.split('%')[0])
    # gpu_util is something like: 0
    return gpu_util


def preprocess_df_Rehuohra(path_to_csv, path_to_data, colour_band, file_extension):
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

    col_list = ['field parcel', 'PLANT', 'full crop loss scaled', 'partial crop loss scaled']

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


def preprocess_df_top4(path_to_csv, path_to_data, colour_band, file_extension, balanced):
    # load data frame which contains field parcels and locations
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
    print(df.shape[0] - len(np.unique(df['field parcel'])), 'duplicate entries')
    df = df[df['field parcel'].duplicated() == False]
    print(df.shape[0] - len(np.unique(df['field parcel'])), 'duplicate entries')

    print('create new column: relative crop loss = crop loss / Property area')
    df['full crop loss scaled'] = df['full crop loss'] / df['Property area']
    df['partial crop loss scaled'] = df['partial crop loss'] / df['Property area']

    print('categorise the data')

    i_full = df[(df['full crop loss scaled'] == 1) & (df['partial crop loss scaled'] == 0)].index
    df.loc[i_full, 'category'] = 'full'
    del i_full

    i_partial = df[(df['partial crop loss scaled'] == 1) & (df['full crop loss scaled'] <= 0)].index
    df.loc[i_partial, 'category'] = 'partial'
    del i_partial

    i_noloss = df[(df['full crop loss scaled'] == 0) & (df['partial crop loss scaled'] == 0)].index
    df.loc[i_noloss, 'category'] = 'noloss'
    del i_noloss

    i_fullandpartial = df[
        (df['full crop loss scaled'] > 0) & (df['full crop loss scaled'] < 1) &
        (df['partial crop loss scaled'] > 0) & (df['partial crop loss scaled'] < 1)
        ].index
    df.loc[i_fullandpartial, 'category'] = 'fullandpartial'
    del i_fullandpartial

    threshold = 0.5
    i_anyloss = df[
        (df['full crop loss scaled'] > threshold) | (df['partial crop loss scaled'] > threshold)
        ].index
    df.loc[i_anyloss, 'category_2'] = 'anyloss'
    del i_anyloss

    # count how many instances of each category each plant has
    plants = []
    numbers = []
    fulls, partials, nolosses, fullandpartials = [], [], [], []
    anylosses = []

    for plant in np.unique(df['PLANT']):
        plants.append(plant)
        df_tmp = df[df['PLANT'] == plant]
        numbers.append(len(df_tmp))
        fulls.append(len(df_tmp[df_tmp['category'] == 'full']))
        partials.append(len(df_tmp[df_tmp['category'] == 'partial']))
        nolosses.append(len(df_tmp[df_tmp['category'] == 'noloss']))
        fullandpartials.append(len(df_tmp[df_tmp['category'] == 'fullandpartial']))
        anylosses.append(len(df_tmp[df_tmp['category_2'] == 'anyloss']))
        del df_tmp

    df_numbers = pd.DataFrame(
        {'plant': plants,
         'number': numbers,
         'full': fulls,
         'partial': partials,
         'noloss': nolosses,
         'fullandpartial': fullandpartials,
         'anyloss': anylosses
         })
    del fulls, partials, nolosses, fullandpartials, anylosses, plants, numbers
    df_numbers = df_numbers.sort_values(by=['anyloss'], ascending=False)

    n = 5
    print(f'select top {n} plants according to number of fields with any loss')
    top_5_plants = df_numbers.sort_values(by=['anyloss'], ascending=False).plant[0:n].values

    print('drop Rehuohra because it is already downloaded')
    top_5_plants = np.delete(top_5_plants, np.where(top_5_plants == 'Rehuohra'))
    print(top_5_plants)

    print('reduce df to selected plants')
    df = df[(df.PLANT == top_5_plants[0]) | (df.PLANT == top_5_plants[1]) | (df.PLANT == top_5_plants[2]) | (
            df.PLANT == top_5_plants[3])]

    print('reduce df to only those fields, which are either full, partial, fullandpartial or anyloss '
          'and select the largest 6k fields with no loss')

    i_losses, i_noloss6ks = [], []
    for plant in top_5_plants:
        df_plant = df[df.PLANT == plant]
        i_loss = df_plant[(df_plant.category == 'full') | (df_plant.category == 'partial') | (
                df_plant.category == 'fullandpartial') | (df_plant.category_2 == 'anyloss')].index.values.tolist()
        i_losses += i_loss
        if balanced:
            n_noloss = len(i_loss)
        else:
            n_noloss = 6000
        i_noloss6ks += df_plant[
                           (df_plant.category == 'noloss')
                       ].sort_values(by=['Property area'], ascending=False).index[0:n_noloss].values.tolist()

    df = df[(df.index.isin(i_losses)) | df.index.isin(i_noloss6ks)]

    print('check that all the remaining indeces have to be either in i_losses or in i_nolosses6ks')
    if np.any(np.array([df.index.isin(i_losses), df.index.isin(i_noloss6ks)]), axis=0).all():
        print(f'checking file existence for {len(df)} fields for {top_5_plants}')

        col_list = ['field parcel', 'PLANT', 'full crop loss scaled', 'partial crop loss scaled']

        # print('trim data frame to:', col_list)
        df = df[col_list]

        # check if files for fields exist, if not, remove from data frame
        # write relative path including colour band and file extension to dataframe for future usage
        print('total number of fields before verifying file existence ', df.shape[0])
        subfolders = ['dataset9/']
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
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon


def plot_latent_space(model, dataset, steps_per_epoch, batch_size, example_images, ex_im_informations,
                      path='../4_runs/plots/latent/'):
    """Plots labels and satellite images as function of 2-dim latent vector

    # Arguments:
        :param model: tuple of encoder and decoder model
        :param dataset: test dataset
        :param steps_per_epoch: steps per epoch when using model.predict
        :param batch_size: batch size
        :param example_images: path to the example images to display in the 2D latent representation
        :param ex_im_informations: information for the example_images to which class they belong
        :param path: path for saving the plots
    """

    encoder, decoder = model
    latent_dim = encoder.output[0].shape.dims[1].value

    # models need to be compiled before they can be used for .predict()
    rmsprop = tf.keras.optimizers.RMSprop(lr=0.00001)
    encoder.compile(optimizer=rmsprop, loss='mse')
    decoder.compile(optimizer=rmsprop, loss='mse')

    img_size = 512
    n_channels = 13

    os.makedirs(path, exist_ok=True)

    print('display a 2D plot of the satellite images projected into the latent space')
    filename = os.path.join(path, "z_mean_over_latent.png")

    # preventing error when using model.predict to expect also target variables when used with tf.data.Dataset
    # Target = output of the encoder: z_mean, z_log_var, z
    # create empty output dataset for encoder
    output_set = tf.data.Dataset.from_tensor_slices(
        (
            np.zeros(shape=(batch_size * steps_per_epoch, latent_dim), dtype=np.float32),
            np.zeros(shape=(batch_size * steps_per_epoch, latent_dim), dtype=np.float32),
            np.zeros(shape=(batch_size * steps_per_epoch, latent_dim), dtype=np.float32),
        )
    )
    output_set = output_set.batch(batch_size).repeat()

    # Group the input and output dataset
    dataset_m = tf.data.Dataset.zip((dataset, output_set))

    z_mean, _, _ = encoder.predict(dataset_m, verbose=1, steps=steps_per_epoch)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(z_mean[:, 0], z_mean[:, 1], s=3, zorder=1)

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
        im = ski_io.imread(file)

        z_mean, _, _ = encoder.predict(
            np.reshape(im, (1, img_size, img_size, n_channels))
            # reshape necessary because the encoder expects a 4D array with batch x img_size x img_size x n_channels
        )
        z_mean = z_mean[0][[0, 1]]  # select first batch and first 2 dimensions of the latent vector

        ax.scatter(z_mean[0], z_mean[1], s=10)

        # display information tag of the image
        textbox = TextArea(info, minimumdescent=False)

        ab = AnnotationBbox(textbox, z_mean,
                            xybox=(-80, 160),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            )
        ax.add_artist(ab)

        # display the image itself
        im_rgb = im[:, :, [3, 2, 1]]
        im_rgb = exposure.rescale_intensity(im_rgb)
        imagebox = OffsetImage(im_rgb, zoom=0.25)

        ab = AnnotationBbox(imagebox, z_mean,
                            xybox=(-80., 80.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.1,
                            arrowprops=dict(arrowstyle="->"))

        ax.add_artist(ab)
    if latent_dim == 2:
        plt.title('Satellite images projected into the latent space')
    elif latent_dim > 2:
        plt.title('Satellite images projected into the latent space (only first two dimensions of latent space are '
                  'shown)')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close('all')

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
            z_sample = [xi, yi]
            while len(z_sample) < latent_dim:
                z_sample.append(0)
            z_sample = np.array([z_sample])
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
    if latent_dim == 2:
        plt.title('Decoded latent space')
    elif latent_dim > 2:
        plt.title('Decoded latent space (only first two dimensions of latent space are shown)')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close('all')


def _parse_function_2imgs(example_proto):
    feature_description = {
        'image': tf.FixedLenFeature([], tf.string),
    }

    # First: parse the input tf.Example proto using the dictionary above.
    parsed_features = tf.parse_single_example(example_proto, feature_description)

    # Decode saved image string into an array
    image = tf.decode_raw(parsed_features['image'], tf.float32)  # tensor is still flattened
    image = tf.reshape(image, (512, 512, 13))

    # Second: return a tuple of desired variables
    return image, image


def _parse_function_1img(example_proto):
    feature_description = {
        'image': tf.FixedLenFeature([], tf.string),
    }

    # First: parse the input tf.Example proto using the dictionary above.
    parsed_features = tf.parse_single_example(example_proto, feature_description)

    # Decode saved image string into an array
    image = tf.decode_raw(parsed_features['image'], tf.float32)  # tensor is still flattened
    image = tf.reshape(image, (512, 512, 13))

    # Second: return a tuple of desired variables
    return image


def _parse_function_pred(example_proto):
    feature_description = {
        'image': tf.FixedLenFeature([], tf.string),
        'full_crop_loss_value': tf.FixedLenFeature([], tf.float32),
        'partial_crop_loss_value': tf.FixedLenFeature([], tf.float32),
        'image_path': tf.FixedLenFeature([], tf.string, default_value=''),
        'plant': tf.FixedLenFeature([], tf.string, default_value=''),
    }

    # First: parse the input tf.Example proto using the dictionary above.
    parsed_features = tf.parse_single_example(example_proto, feature_description)

    # Decode saved image string into an array
    image = tf.decode_raw(parsed_features['image'], tf.float32)  # tensor is still flattened
    image = tf.reshape(image, (512, 512, 13))

    f_cl = parsed_features['full_crop_loss_value']
    p_cl = parsed_features['partial_crop_loss_value']

    # define 4D and 2D loss category as one-hot encoded array
    loss_cat_4d_one_hot_tensor = tf.cond(
        tf.logical_and(tf.math.greater(f_cl, 0), tf.math.equal(p_cl, 0)),  # only full loss
        lambda: tf.one_hot(1 - 1, depth=4, dtype=tf.int8),
        lambda: tf.cond(
            tf.logical_and(tf.math.equal(f_cl, 0), tf.math.greater(p_cl, 0)),  # only partial loss
            lambda: tf.one_hot(2 - 1, depth=4, dtype=tf.int8),
            lambda: tf.cond(
                tf.logical_and(tf.math.greater(f_cl, 0), tf.math.greater(p_cl, 0)),  # both full and partial loss
                lambda: tf.one_hot(3 - 1, depth=4, dtype=tf.int8),
                lambda: tf.one_hot(4 - 1, depth=4, dtype=tf.int8)  # no loss
            )
        )
    )

    loss_cat_2d_one_hot_tensor = tf.cond(
        tf.logical_or(tf.math.greater(f_cl, 0), tf.math.greater(p_cl, 0)),  # some loss
        lambda: tf.one_hot(1, depth=2, dtype=tf.int8),
        lambda: tf.one_hot(0, depth=2, dtype=tf.int8),  # no loss
    )

    im_path = parsed_features['image_path']
    plant_name = parsed_features['plant']
    # im_path and plant_name are still an encoded tf.Tensor and
    # thus needs to be converted to numpy with .numpy() and then decoded decode('utf-8')

    # define plant type and parse as one-hot encoding
    n_plants = 5
    top_5_plants = ['Rehuohra', 'Kaura', 'Mallasohra', 'Kevätvehnä', 'Kevätrypsi']

    plant_cat_one_hot = tf.cond(
        tf.math.equal(plant_name, top_5_plants[0]), lambda: tf.one_hot(0, depth=n_plants, dtype=tf.int8),
        lambda: tf.cond(
            tf.math.equal(plant_name, top_5_plants[1]), lambda: tf.one_hot(1, depth=n_plants, dtype=tf.int8),
            lambda: tf.cond(
                tf.math.equal(plant_name, top_5_plants[2]), lambda: tf.one_hot(2, depth=n_plants, dtype=tf.int8),
                lambda: tf.cond(
                    tf.math.equal(plant_name, top_5_plants[3]),
                    lambda: tf.one_hot(3, depth=n_plants, dtype=tf.int8),
                    lambda: tf.one_hot(4, depth=n_plants, dtype=tf.int8)
                )
            )
        )
    )

    # Second: return a tuple of desired variables
    return (image), (f_cl, p_cl, loss_cat_4d_one_hot_tensor, loss_cat_2d_one_hot_tensor, plant_cat_one_hot)


def create_tfdataDataset(path, name, prediction, batch_size, prefetch_size, num_parallel_readers):
    """
    inspired from https://www.tensorflow.org/guide/performance/datasets#input_pipeline_structure
    :param path: path to directory where name_[0-9]+.tfrecord are located
    :param name: either train, val or test
    :param prediction: boolean indicating usage for the prediciton model
    :param batch_size:
    :param prefetch_size:
    :param num_parallel_readers:
    :return:
    """

    print(f'reading TFRecords file: {name}')

    # tf_records_filenames = glob.glob(os.path.join(path, name + '_*.tfrecord'))

    files = tf.data.Dataset.list_files(os.path.join(path, name + '_*.tfrecord'))

    f = open(os.path.join(path, 'n_files_' + name + '.txt'), 'r')
    n_files = int(f.read())
    steps_per_epoch = n_files // batch_size

    # Construct a TFRecordDataset
    # dataset = tf.data.TFRecordDataset(tf_records_filenames)
    # dataset = files.interleave(tf.data.TFRecordDataset)

    # better for data stored remotely
    dataset = files.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=num_parallel_readers))

    # Set the number of datapoints you want to load and shuffle
    # dataset = dataset.shuffle(n_files)

    # Maps the parser on every filepath in the array. Set the number of parallel loaders here
    if prediction:
        dataset = dataset.map(_parse_function_pred, num_parallel_calls=os.cpu_count() - 1)
    else:
        if name == 'test':
            dataset = dataset.map(_parse_function_1img, num_parallel_calls=os.cpu_count() - 1)
        else:
            dataset = dataset.map(_parse_function_2imgs, num_parallel_calls=os.cpu_count() - 1)
    dataset = dataset.batch(batch_size=batch_size)

    # replaces .map and .batch -- but DEPRECATED??
    # dataset = dataset.apply(tf.data.experimental.map_and_batch(
    #     map_func=_parse_function, batch_size=batch_size))

    dataset = dataset.prefetch(prefetch_size)

    # This dataset will go on forever
    dataset = dataset.repeat()

    return dataset, steps_per_epoch


def plot_confusion_matrix(y_true, y_pred, class_names, path, file_name_prefix='', normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    else:
        print(title)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # class_names = class_names[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if normalize:
        plt.savefig(os.path.join(path, file_name_prefix + 'confusion_matrix_norm.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(path, file_name_prefix + 'confusion_matrix.png'), dpi=300, bbox_inches='tight')

    plt.close('all')

    return


def f1(y_true, y_pred):
    """define f1 score as custom metric for categorical binary data"""
    y_true = tf.argmax(y_true, axis=1)  # convert probabilities to category
    y_pred = tf.argmax(y_pred, axis=1)  # convert probabilities to category
    y_pred = tf.keras.backend.round(y_pred)
    tp = tf.keras.backend.sum(tf.keras.backend.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = tf.keras.backend.sum(tf.keras.backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = tf.keras.backend.sum(tf.keras.backend.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + tf.keras.backend.epsilon())
    r = tp / (tp + fn + tf.keras.backend.epsilon())

    f1_score = 2 * p * r / (p + r + tf.keras.backend.epsilon())
    f1_score = tf.where(tf.is_nan(f1_score), tf.zeros_like(f1_score), f1_score)
    return tf.keras.backend.mean(f1_score)


def my_acc(y_true, y_pred):
    """ calculate the accuracy for one-hot encoded labels and predictions"""
    y_true = tf.argmax(y_true, axis=1)  # convert probabilities to category
    y_pred = tf.argmax(y_pred, axis=1)  # convert probabilities to category
    y_pred = tf.keras.backend.round(y_pred)

    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred), axis=-1)
