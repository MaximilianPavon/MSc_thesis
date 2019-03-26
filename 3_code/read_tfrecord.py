import tensorflow as tf
import argparse
import sys
import os
import numpy as np
from skimage import io
import glob
tf.enable_eager_execution()


def _parse_function(example_proto):
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
        tf.logical_and(tf.math.greater(f_cl, 0), tf.math.equal(p_cl, 0)),            # only full loss
        lambda: tf.one_hot(1 - 1, depth=4, dtype=tf.int8),
        lambda: tf.cond(
            tf.logical_and(tf.math.equal(f_cl, 0), tf.math.greater(p_cl, 0)),        # only partial loss
            lambda: tf.one_hot(2 - 1, depth=4, dtype=tf.int8),
            lambda: tf.cond(
                tf.logical_and(tf.math.greater(f_cl, 0), tf.math.greater(p_cl, 0)),  # both full and partial loss
                lambda: tf.one_hot(3 - 1, depth=4, dtype=tf.int8),
                lambda: tf.one_hot(4 - 1, depth=4, dtype=tf.int8)                                 # no loss
            )
        )
    )

    loss_cat_2d_one_hot_tensor = tf.cond(
        tf.logical_or(tf.math.greater(f_cl, 0), tf.math.greater(p_cl, 0)),  # some loss
        lambda: tf.one_hot(1, depth=2, dtype=tf.int8),
        lambda: tf.one_hot(0, depth=2, dtype=tf.int8),                                   # no loss
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
    return image, image, f_cl, p_cl, im_path, plant_name, loss_cat_4d_one_hot_tensor, loss_cat_2d_one_hot_tensor, plant_cat_one_hot


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project_path", help="Specify project path, where the project is located.")
    parser.add_argument("-d", "--data_path",
                        help="Specify path, where the data is located. E.g. /tmp/$SLURM_JOB_ID/05_images_masked/ ")

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

    # Parameters
    path_to_data = args.data_path if args.data_path else os.path.join(args.project_path, '2_data/03_images_subset_masked/')

    for f_name in ['train', 'val', 'test']:

        print(f'reading TFRecords file: {f_name}')

        tf_records_filenames = glob.glob(os.path.join(path_to_data, f_name + '_*.tfrecord'))

        parsed_dataset = tf.data.TFRecordDataset(tf_records_filenames).map(
            _parse_function, num_parallel_calls=os.cpu_count()).shuffle(100)

        for im1, im2, f_cl, p_cl, im_path, plant_name, loss_cat_4d, loss_cat_2d, plant_cat in parsed_dataset.take(10):

            print('im1.shape:', im1.shape)
            print('im2.shape:', im2.shape)
            print(f_cl)
            print(p_cl)

            print(plant_name)
            print(plant_name.numpy().decode('utf-8'))
            print(plant_cat)

            print(loss_cat_4d)
            print(loss_cat_4d.numpy())
            print(loss_cat_2d)
            print(loss_cat_2d.numpy())

            print(im_path)

            im_path = im_path.numpy().decode('utf-8')
            im_path = os.path.join(args.project_path, im_path)

            im_loaded = io.imread(im_path)
            print(np.array_equal(im1.numpy(), im_loaded))

            print()
