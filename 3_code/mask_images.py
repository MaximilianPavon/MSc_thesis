import geopandas as gpd
import pandas as pd
import argparse
import os
import glob
import rasterio.mask
from tqdm import tqdm
import numpy as np
import sys


def get_field_id(file_paths):
    assert isinstance(file_paths, (list,))
    field_id = [i.split('/')[-1] for i in file_paths]
    field_id = [i.split('_')[0] for i in field_id]
    return field_id


def get_output_path(input_path):
    split_path = input_path.split('/')
    if args.computer == 'workstation':
        split_path[6] = '05_images_masked'
    else:
        split_path[5] = '05_images_masked'
    output_path = '/'.join(split_path)
    os.makedirs('/'.join(split_path[0:-1]), exist_ok=True)
    return output_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group(title='required arguments')
    req_grp.add_argument("-c", "--computer", help="Specify computer: use \'triton\', \'mac\' or \'workstation\'.", required=True)
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

    # load shape file containing the field parcel geometries
    shape_file = os.path.join(args.project_path, '2_data/04_small_data/rap_2015_rehuohra_shp/rap_2015_rehuohra.shp')
    vector_df = gpd.read_file(shape_file)

    # get path to all the images
    df = pd.read_csv(os.path.join(args.project_path, '2_data/04_small_data/fof_train_triton.csv'))
    images = [os.path.join(args.project_path, i) for i in df['triton_path'].tolist()]

    # images = glob.glob(os.path.join(args.project_path, '2_data/04_small_data/noloss/good/*.tiff'))
    # images += glob.glob(os.path.join(args.project_path, '2_data/04_small_data/noloss/not_good/*.tiff'))
    # images += glob.glob(os.path.join(args.project_path, '2_data/04_small_data/fullandpartial/*.tiff'))
    nimages = len(images)

    print('{0:7d} images'.format(nimages))

    not_found = []

    for i in tqdm(range(nimages)):
        image_path = images[i]
        field_id = get_field_id([image_path])[0]
        field_poly = vector_df.loc[vector_df.field_parc == field_id]['geometry']

        if not field_poly.is_valid.values:
            not_found.append(image_path)
            continue

        # open image of the field, apply mask and save it to disk
        with rasterio.open(image_path) as src:
            out_im, out_transform = rasterio.mask.mask(src, field_poly, crop=False)
            out_kwargs = src.profile

        out_file = get_output_path(image_path)

        with rasterio.open(out_file, "w", **out_kwargs) as dest:
            dest.write(out_im)
        # print('images written: {0}'.format(i))

    np.savetxt(
        os.path.join(args.project_path, '2_data/05_images_masked/', 'not_found.csv'),
        np.array(not_found),
        fmt='%s'
    )

    print('done')
