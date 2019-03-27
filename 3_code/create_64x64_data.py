import argparse, sys, os
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project_path", help="Specify project path, where the project is located.")
    req_grp = parser.add_argument_group(title='required arguments')
    req_grp.add_argument("-c", "--computer", help="Specify computer: use \'triton\', \'mac\' or \'workstation\'.",
                         required=True)
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

    path_to_csv = os.path.join(args.project_path, '2_data/01_MAVI_unzipped_preprocessed/2015/pp_balanced_top5.csv')
    path_to_data = os.path.join(args.project_path, '2_data/03_images_subset_masked/')

    df = pd.read_csv(path_to_csv)

    # create 4d loss category
    df.loc[df[(df['full crop loss scaled'] > 0) & (df['partial crop loss scaled'] == 0)].index, 'loss_cat_4d'] = 1  # full
    df.loc[df[(df['full crop loss scaled'] == 0) & (df['partial crop loss scaled'] > 0)].index, 'loss_cat_4d'] = 2  # partial
    df.loc[df[(df['full crop loss scaled'] > 0) & (df['partial crop loss scaled'] > 0)].index, 'loss_cat_4d'] = 3  # full and partial
    df.loc[df[(df['full crop loss scaled'] == 0) & (df['partial crop loss scaled'] == 0)].index, 'loss_cat_4d'] = 4  # no loss
    df['loss_cat_4d'] = df['loss_cat_4d'].astype(int)

    # create 2d loss category
    df.loc[df[(df['full crop loss scaled'] == 0) & (df['partial crop loss scaled'] == 0)].index, 'loss_cat_2d'] = 0  # no loss
    df.loc[df[(df['full crop loss scaled'] > 0) | (df['partial crop loss scaled'] > 0)].index, 'loss_cat_2d'] = 1  # some loss
    df['loss_cat_2d'] = df['loss_cat_2d'].astype(int)

    # create plant category

    top_5_plants = ['Rehuohra', 'Kaura', 'Mallasohra', 'Kevätvehnä', 'Kevätrypsi']  # in this order
    for i, plant in enumerate(top_5_plants):
        df.loc[df[(df['PLANT'] == plant)].index, 'plant_cat'] = i
    df['plant_cat'] = df['plant_cat'].astype(int)

    # List of image paths, np array of labels
    IDs = df['field parcel'].values
    im_paths = [os.path.join(path_to_data, v) for v in df['partial path'].tolist()]
    full_cl_values = df['full crop loss scaled'].values
    partial_cl_values = df['partial crop loss scaled'].values
    loss_categories_4d = df['loss_cat_4d'].values
    loss_categories_2d = df['loss_cat_2d'].values
    plant_categories = df['plant_cat'].values

    X = []
    for im_path in tqdm(im_paths):

        image = io.imread(im_path)
        image = image[::8, ::8, :]  # downsample from 512x512 to 64x64

        # compute NDVI value for the image from the 13 channels
        # NDVI = = (B08 - B04) / (B08 + B04)
        # raw channels are: [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12]

        check = image[:, :, 7] - image[:, :, 3] != 0
        image_ndvi = np.where(check, (image[:, :, 7] - image[:, :, 3]) / (image[:, :, 7] - image[:, :, 3]), 0)
        image_ndvi = np.reshape(image_ndvi, (64, 64, 1))

        X.append(image_ndvi)

    X = np.array(X)
    y = np.array([full_cl_values, partial_cl_values, loss_categories_4d, loss_categories_2d, plant_categories])
    print()

    np.save(os.path.join(args.project_path, '2_data/04_toydata_64x64', 'X64_top5.npy'), X)
    np.save(os.path.join(args.project_path, '2_data/04_toydata_64x64', 'Y64_top5.npy'), y)
    np.savetxt(os.path.join(args.project_path, '2_data/04_toydata_64x64', 'ID64_top5.txt'), IDs, fmt='%s')
    print()
