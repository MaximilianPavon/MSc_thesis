import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentinelhub import WmsRequest, MimeType, CRS, BBox, CustomUrlParam


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


def save_images_for_df(df, path_to_data, layers, max_cc, time_window):
    """
    goes over data frame and choose parcel field as name and extract location for WMSRequest

    :param df: data fram to be looped over
    :param path_to_data: path where to save images
    :param layers: layers to be downloaded in the WMSRequest
    :param max_cc: maximum accepted cloud coverage of an image
    :param time_window: time window (start date, end date)
    :return:
    """

    n_fields = df.shape[0]  # total number of fields
    size = 512

    c = 0

    with tqdm(total=n_fields*len(layers)) as pbar:

        for index, row in df.iterrows():

            for layer in layers:

                if c < 186062:
                    pbar.update(1)
                    c=c+1
                    continue

                # get parcel field for naming the image
                parcel_field = str(row['field parcel'])

                # get coordinates of bounding box
                upper_left = [row['xmin'] / 10 ** 4, row['ymax'] / 10 ** 5]
                lower_right = [row['xmax'] / 10 ** 4, row['ymin'] / 10 ** 5]

                # create BBox object with previous coordinates
                bbox = create_bbox_coordinates(upper_left, lower_right)

                # depending on the selected layer the file format has to be changed
                if layer == 'RAW':  # change image format since all Sentinel-2’s 13 bands cannot be packed into a png
                    #  image
                    image_format = MimeType.TIFF_d32f
                    file_extension = '.tiff'
                else:
                    image_format = MimeType.PNG
                    file_extension = '.png'

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
                                         custom_url_params={CustomUrlParam.ATMFILTER: 'ATMCOR'}
                                         )

                # get data and simultaneously save the data
                wms_request.get_data(save_data=True)

                # remove additional images, in case more than one image was downloaded and rename the last file
                # original naming would be: wms_TRUE-COLOR-S2-L1C_EPSG4326_46.16_-16.15_46.51_-15.58_2017-12-15T07-12
                # -03_512X849.png
                n_images = len(wms_request.get_filename_list())
                for i in range(n_images):
                    if i + 1 < n_images:
                        os.remove(path_to_data + wms_request.get_filename_list()[i])
                    elif i + 1 == n_images:

                        os.rename(path_to_data + wms_request.get_filename_list()[i],
                                  path_to_data + parcel_field + '_' + layer + file_extension)

                c = c + 1
                pbar.update(1)


if __name__ == '__main__':

    # set some global variables
    INSTANCE_ID = '02a54b79-4c74-4960-a377-70c15518221b'
    path_to_csv = '/Users/maximilianproll/Dropbox (Aalto)/2_data/01_MAVI_unzipped_preprocessed/MAVI2/2015/rap_2015.csv'
    layers = ['NDVI', 'RAW']
    path_to_data = '/Users/maximilianproll/Dropbox (Aalto)/2_data/02_data_incorrect_coord/'
    time_window = ('2015-08-01', '2015-08-31')
    max_cc = 0.3

    # load data frame which contains field parcels and locations
    df = pd.read_csv(path_to_csv)
    print('successfully loaded file: ', path_to_csv)

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
    fieldparcel = df['field parcel']
    df = df[fieldparcel.duplicated() == False]
    print(df.shape[0] - len(np.unique(df['field parcel'])), 'duplicate entries')

    print('total number of fields: ', df.shape[0])

    # select only those rows, where both full and partial crop loss are present
    # print('only ', len(df['full crop loss'].dropna()), '(full) resp. ', len(df['partial crop loss'].dropna()),
    #       '(partial) of a total ', len(df), 'have a crop loss information')
    # df = df.dropna(subset=['full crop loss', 'partial crop loss'])
    # print('total number of fields where both full and partial crop loss are present : ', len(df))

    # select largest number of samples for one given plant species
    plants = df['PLANT']
    num = 0
    for plant in np.unique(list(plants)):
        num_tmp = len(df[plants == plant])
        # print(plant, '\t ', num_tmp)

        if num_tmp > num:
            num = num_tmp
            plant_max = plant
    print('maximum number for', plant_max, 'with', num, 'entries')
    df = df[plants == plant_max]

    save_images_for_df(df, path_to_data, layers, max_cc, time_window)
