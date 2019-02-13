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


def serialize_example(image_string, full_crop_loss_label, partial_crop_loss_label, path):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
        'image': _bytes_feature(image_string),
        'full_crop_loss_label': _float_feature(full_crop_loss_label),
        'partial_crop_loss_label': _float_feature(partial_crop_loss_label),
        'image_path': _bytes_feature(path),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project_path", help="Specify project path, where the project is located.")
    parser.add_argument("-d", "--data_path",
                        help="Specify path, where the data is located. E.g. /tmp/$SLURM_JOB_ID/05_images_masked/ ")
    parser.add_argument("--file_limit", type=float, default=4.0 ,  help="Define maximum file size in GB of .tfrecords files. Default 4 GB")
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
        'path_to_csv': os.path.join(args.project_path,'2_data/01_MAVI_unzipped_preprocessed/MAVI2/2015/preprocessed_masked.csv'),
        'train_p': 0.8,
        'val_p': 0.1,
        'path_to_data': args.data_path if args.data_path else os.path.join(args.project_path, '2_data/05_images_masked/'),
    }

    df = pd.read_csv(params['path_to_csv'])

    # split data frame into train, validation and test
    train_df, val_df, test_df = split_dataframe(df, params['train_p'], params['val_p'], )
    del df

    for f_name, data_frame in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):

        file_counter = 0

        print(f'creating TFRecords file: {f_name}')

        # List of image paths, np array of labels
        im_list = [os.path.join(params['path_to_data'], v) for v in data_frame.index.tolist()]
        f_cl_labels_arr = data_frame['full crop loss scaled'].values
        p_cl_labels_arr = data_frame['partial crop loss scaled'].values

        # create writer object
        tf_records_filename = os.path.join(params['path_to_data'], f_name + '_' + str(file_counter) + '.tfrecord')
        writer = tf.python_io.TFRecordWriter(tf_records_filename)

        # Loop over images and labels, wrap in TF Examples, write away to TFRecord file
        for i in tqdm(range(len(data_frame)), total=len(data_frame)):
            f_cl_label = f_cl_labels_arr[i].astype(np.float32)
            p_cl_label = p_cl_labels_arr[i].astype(np.float32)

            image = io.imread(im_list[i])

            # serialize the bytes representation of the image, the crop los values as well as the image path
            example_serialized = serialize_example(image.tostring(), f_cl_label, p_cl_label, im_list[i].encode('utf-8'))

            file_size = os.stat(tf_records_filename).st_size / (1024 ** 3)  # convert file size in bytes to GB

            if file_size > args.file_limit:
                writer.close()  # close old file writer
                file_counter += 1  # increase file counter by one and create new file writer
                tf_records_filename = os.path.join(params['path_to_data'],
                                                   f_name + '_' + str(file_counter) + '.tfrecord')
                writer = tf.python_io.TFRecordWriter(tf_records_filename)

            writer.write(example_serialized)

        writer.close()
