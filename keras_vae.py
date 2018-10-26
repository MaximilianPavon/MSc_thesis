import pandas as pd
import numpy as np
import os.path
from my_classes import DataGenerator
from keras.preprocessing.image import ImageDataGenerator


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


def split_dataframe(df, train_p, val_p, random_state=200):
    # split data frame into train, validation and test
    train_df = df.sample(frac=train_p, random_state=random_state)
    df = df.drop(train_df.index)
    val_df = df.sample(frac=val_p / (1 - train_p), random_state=random_state)
    test_df = df.drop(val_df.index)

    # make field parcel as indeces
    train_df = train_df.set_index('field parcel')
    val_df = val_df.set_index('field parcel')
    test_df = test_df.set_index('field parcel')

    return train_df, val_df, test_df


if __name__ == '__main__':
    path_to_csv = 'MAVI2/2015/rap_2015.csv'
    path_to_data = 'data/'
    file_extension = '.tiff'
    colour_band = 'RAW'
    train_p, val_p = 0.8, 0.1

    df = preprocess_df(path_to_csv, path_to_data, colour_band,  file_extension)

    # add color spectrum and file extension to field parcel column in order to match to the file name
    # df['field parcel'] = df['field parcel'] + '_' + color_spectrum

    # split data frame into train, validation and test
    train_df, val_df, test_df = split_dataframe(df, train_p, val_p,)
    del df

    # create dictionaries of IDs and labels
    partition = {
        'train': train_df.index.tolist(), 'validation': val_df.index.tolist(), 'test': test_df.index.tolist()
    }
    labels = {**train_df['full crop loss scaled'].to_dict(), **val_df['full crop loss scaled'].to_dict(),
              **test_df['full crop loss scaled'].to_dict()}

    # Parameters
    params = {
        'path_to_data': path_to_data,
        'colour_band': colour_band,
        'file_extension': file_extension,
        'dim': (512, 512),
        'batch_size': 64,
        'n_channels': 13,
        'shuffle': True}

    # Generators
    training_generator = DataGenerator(partition['train'], labels,  **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)

    i = 0
    for x, y in training_generator:
        print('x ', x.shape,'y ', y.shape)
        if i > 10:
            break
        i += 1