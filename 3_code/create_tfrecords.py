import argparse
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io
from tqdm import tqdm
from my_functions import split_dataframe


# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image_string, full_crop_loss_value, partial_crop_loss_value, path, plant_name):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
        'image': _bytes_feature(image_string),
        'full_crop_loss_value': _float_feature(full_crop_loss_value),
        'partial_crop_loss_value': _float_feature(partial_crop_loss_value),
        'image_path': _bytes_feature(path),
        'plant': _bytes_feature(plant_name),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project_path", help="Specify project path, where the project is located.")
    parser.add_argument("-d", "--data_path", help="Specify path, where the data is located.")
    parser.add_argument("--file_limit", type=float, default=6.0 ,  help="Define maximum file size in GB of .tfrecords files. Default 6 GB")
    req_grp = parser.add_argument_group(title='required arguments')
    req_grp.add_argument("-c", "--computer", help="Specify computer: use \'triton\', \'mac\' or \'workstation\'.", required=True)
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

    # Parameters
    params = {
        'path_to_csv': os.path.join(args.project_path,'2_data/01_MAVI_unzipped_preprocessed/2015/pp_balanced_top5.csv'),
        'train_p': 0.8,
        'val_p': 0.1,
        'path_to_data': args.data_path if args.data_path else os.path.join(args.project_path, '2_data/03_images_subset_masked/'),
    }

    df = pd.read_csv(params['path_to_csv'])

    # split data frame into train, validation and test
    train_df, val_df, test_df = split_dataframe(df, params['train_p'], params['val_p'], )
    del df

    for f_name, data_frame in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):

        file_counter = 0

        print(f'creating TFRecords file: {f_name}')

        # List of image paths, np array of labels
        im_paths = [os.path.join('2_data/03_images_subset_masked/', v) for v in data_frame['partial path'].tolist()]
        full_cl_values = np.clip(data_frame['full crop loss scaled'].values, 0, 1)
        partial_cl_values = np.clip(data_frame['partial crop loss scaled'].values, 0, 1)
        plant_names = data_frame['PLANT'].values

        # create writer object
        tf_records_filename = os.path.join(params['path_to_data'], f_name + '_' + str(file_counter) + '.tfrecord')
        writer = tf.python_io.TFRecordWriter(tf_records_filename)

        n_files = len(data_frame)
        outF = open(os.path.join(params['path_to_data'], 'n_files_' + f_name + '.txt'), 'w')
        outF.write(str(n_files))
        outF.close()

        # Loop over images and labels, wrap in TF Examples, write away to TFRecord file
        for im_path, full_cl, partial_cl, plant_name in tqdm(zip(im_paths, full_cl_values, partial_cl_values, plant_names), total=n_files):
            full_cl = full_cl.astype(np.float32)
            partial_cl = partial_cl.astype(np.float32)

            image = io.imread(os.path.join(args.project_path, im_path))

            # serialize the bytes representation of the image, the crop los values as well as the image path
            example_serialized = serialize_example(image.tostring(), full_cl, partial_cl, im_path.encode('utf-8'), plant_name.encode('utf-8'))

            file_size = os.stat(tf_records_filename).st_size / (1024 ** 3)  # convert file size in bytes to GB

            if file_size > args.file_limit:
                writer.close()  # close old file writer
                file_counter += 1  # increase file counter by one and create new file writer
                tf_records_filename = os.path.join(params['path_to_data'],
                                                   f_name + '_' + str(file_counter) + '.tfrecord')
                writer = tf.python_io.TFRecordWriter(tf_records_filename)

            writer.write(example_serialized)

        writer.close()
