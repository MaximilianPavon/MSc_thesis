import argparse
import os
import numpy as np
import pandas as pd
import sys
from sentinelhub import WmsRequest, MimeType, CRS, BBox, CustomUrlParam
from tqdm import tqdm


def create_bbox_coordinates(upper_left_corner, lower_right_corner):
    """
    :param upper_left_corner: longitude and latitude of upper left corner
    :param lower_right_corner: longitude and latitude of lower right corner
    :return: bbox object, for later use in WMSRequest
    """

    coordinates = []

    for coord in upper_left_corner:
        coordinates.append(coord)

    for coord in lower_right_corner:
        coordinates.append(coord)

    return BBox(bbox=coordinates, crs=CRS.UTM_35N)


def save_images_for_df(df, path_to_data, layers, max_cc_list, time_window, INSTANCE_ID):
    """
    goes over data frame and choose parcel field as name and extract location for WMSRequest

    :param df: data fram to be looped over
    :param path_to_data: path where to save images
    :param layers: layers to be downloaded in the WMSRequest
    :param max_cc_list: list of maximum accepted cloud coverages of an image
    :param time_window: time window (start date, end date)
    :param INSTANCE_ID: alpha-numeric code of length 36, unique for the Sentinel Hub Account
    :return:
    """

    os.makedirs(path_to_data, exist_ok=True)

    n_fields = df.shape[0]  # total number of fields
    size = 512
    not_found_rows = []
    df_not_found_rows = None

    if len(not_found_rows) > 0:
        df = df_not_found_rows

    df = df.reset_index()

    for index, row in tqdm(df.iterrows(), total=n_fields):
        for layer in layers:
            # get parcel field for naming the image
            parcel_field = str(row['field parcel'])

            if os.path.isfile(path_to_data + parcel_field + '_' + layer + ".tiff"):
                print("Already exists [" + str(int(index) + 1) + "/" + str(
                    n_fields) + "]: " + path_to_data + parcel_field + '_' + layer)
            else:
                # notify which one was being downloaded:
                # print("Currently downloading [" + str(int(index) + 1) + "/" + str(n_fields) + "]: " + path_to_data + parcel_field + '_' + layer)

                # get coordinates of bounding box
                upper_left = [row['xmin'], row['ymax']]
                lower_right = [row['xmax'], row['ymin']]

                # create BBox object with previous coordinates
                bbox = create_bbox_coordinates(upper_left, lower_right)

                #  image format
                image_format = MimeType.TIFF_d32f
                file_extension = '.tiff'

                # check over all supplied cc values
                for max_cc in max_cc_list:
                    # create WMS request
                    wms_request = WmsRequest(layer=layer,
                                             bbox=bbox,
                                             # bounding box for the WMS request format: longitude and
                                             # latitude coordinates of upper left and lower right corners
                                             time=time_window,
                                             # acquisition date: '2017-12-15', 'latest' or time window (start
                                             # date, end date)
                                             width=size, height=size,
                                             # It is required to specify at least one of
                                             # `width` and `height` parameters. If only one of them is specified
                                             # the the other one will be calculated to best fit the bounding box
                                             # ratio. If both of them are specified they will be used no matter
                                             # the bounding box ratio.
                                             image_format=image_format,
                                             maxcc=max_cc,
                                             # maximum accepted cloud coverage of an image
                                             data_folder=path_to_data,
                                             # specify folder where to save the downloaded data
                                             instance_id=INSTANCE_ID,
                                             custom_url_params={CustomUrlParam.ATMFILTER: 'ATMCOR'})

                    n_images = len(wms_request.get_filename_list())  # n_images found, only last is actually downloaded
                    # if all records were found with present cc, no need to check other cc values
                    if n_images > 0:  # only download images if images were found
                        # get data and simultaneously save the data
                        wms_request.get_data(save_data=True, data_filter=[-1], max_threads=os.cpu_count(), raise_download_errors=False)
                        break

                if n_images < 1:
                    not_found_rows.append(row.tolist())
                else:
                    # removing additional images, is not required anymore since only the last images will be downloaded
                    # original naming would be: wms_TRUE-COLOR-S2-L1C_EPSG4326_46.16_-16.15_46.51_-15.58_2017-12-15T07-12-03_512X849.png
                    os.rename(path_to_data + wms_request.get_filename_list()[-1],
                              path_to_data + parcel_field + '_' + layer + file_extension)

    # create a dataframe of not found rows
    df_not_found_rows = pd.DataFrame(not_found_rows, columns=list(df.columns.values))

    # save the rows that were not found in any cc value
    df_not_found_rows.to_csv(path_to_data + r"\not_found_rows.csv", sep=',', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group(title='required arguments')
    req_grp.add_argument("-c", "--computer", help="Specify computer: use \'triton\', \'mac\' or \'workstation\'.",
                         required=True)
    parser.add_argument("-p", "--project_path", help="Specify project path, where the project is located.")
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

    # set global variables
    INSTANCE_ID = 'cb23d684-de6f-42e7-9010-57d488cccbc3'
    path_to_csv = os.path.join(args.project_path, '2_data/01_MAVI_unzipped_preprocessed/2015/rap_2015.csv')
    layers = ['BANDS-S2-L1C']
    output_path = os.path.join(args.project_path, '2_data/02_images_original/dataset9/')
    time_window = ('2015-08-01', '2015-08-31')
    max_cc_list = [0.2]

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
        i_losses += df_plant[(df_plant.category == 'full') | (df_plant.category == 'partial') | (
                df_plant.category == 'fullandpartial') | (df_plant.category_2 == 'anyloss')].index.values.tolist()
        i_noloss6ks += df_plant[
                           (df_plant.category == 'noloss')
                       ].sort_values(by=['Property area'], ascending=False).index[0:6000].values.tolist()

    df = df[(df.index.isin(i_losses)) | df.index.isin(i_noloss6ks)]

    print('check that all the remaining indeces have to be either in i_losses or in i_nolosses6ks')
    if np.any(np.array([df.index.isin(i_losses), df.index.isin(i_noloss6ks)]), axis=0).all():
        print(f'starting downloading {len(df)} fields for {top_5_plants}')
        save_images_for_df(df, output_path, layers, max_cc_list, time_window, INSTANCE_ID)
        print('download complete')
