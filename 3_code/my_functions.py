import pandas as pd
import numpy as np
import os.path
from keras import backend as K


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


def sampling(_args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    reparameterization trick
    instead of sampling from Q(z|X), sample eps = N(0,I)
    then z = z_mean + sqrt(var)*eps

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
