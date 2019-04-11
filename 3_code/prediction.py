import argparse
import datetime
import os
import sys

import numpy as np
import tensorflow as tf

from my_functions import get_available_gpus, get_OS, create_tfdataDataset, f1, plot_confusion_matrix

op_sys = get_OS()
if op_sys == 'Darwin':
    # fix for macOS issue regarding library import error:
    # OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    req_grp = parser.add_argument_group(title='required arguments')
    req_grp.add_argument("-c", "--computer", help="Specify computer: use \'triton\', \'mac\' or \'workstation\'.",
                         required=True)
    req_grp.add_argument("-m", "--model", required=True,
                         help="Load trained and compiled models (.hdf5 file) saved by model.save(filepath). "
                              "Path until parent directory: e.g \'4_runs/logging/models/")
    parser.add_argument("-p", "--project_path", help="Specify project path, where the project is located.")
    parser.add_argument("-d", "--data_path",
                        help="Specify path, where the data is located. E.g. /tmp/$SLURM_JOB_ID/05_images_masked/ ")
    parser.add_argument('-e', "--epochs", type=int, help="Specify the number of training epochs")
    parser.add_argument("--batch_normalization", action='store_true', default=False,
                        help="Specify if batch normalizations shall be applied. Default False.")
    parser.add_argument("--param_alternation", type=str,
                        help="When looping over certain hyper parameters, this flag adds the hyper parameter of "
                             "interest to the fron of the hyper parameter string.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode - reduces epochs and steps_per_epoch")
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

    if args.model and not args.project_path in args.model:

        if '..' in args.model:
            # remove leading '..'
            args.model = '/'.join(args.model.split('/')[1:])

        args.model = os.path.join(args.project_path, args.model)

    print('available GPUs:')
    get_available_gpus()

    # Parameters
    path_to_data = args.data_path if args.data_path else os.path.join(args.project_path, '2_data/03_images_subset_masked/')
    im_dim = (512, 512)
    n_channels = 13
    input_shape = (im_dim[0], im_dim[1], n_channels)
    epochs = args.epochs if args.epochs else 200
    batch_normalization = args.batch_normalization
    use_bias = not batch_normalization  # if batch_normalization is used, a bias term can be omitted
    batch_size = 64
    n_parallel_readers = 4

    n_loss_cat_4d = 4
    n_loss_cat_2d = 2
    n_plant_cat = 5

    # create Dataset objects
    ds_train, steps_per_epoch_train = create_tfdataDataset(path_to_data, 'train', True, batch_size, batch_size, n_parallel_readers)
    ds_val, steps_per_epoch_val = create_tfdataDataset(path_to_data, 'val', True, batch_size, batch_size, n_parallel_readers)
    ds_test, steps_per_epoch_test = create_tfdataDataset(path_to_data, 'test', True, batch_size, batch_size, n_parallel_readers)

    print(f'load trained encoder from {args.model}')
    encoder = tf.keras.models.load_model(os.path.join(args.model, 'encoder.hdf5'))
    encoder.trainable = False
    latent_dim = encoder.output[0].shape.dims[1].value

    # folder extension for bookkeeping
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hparam_str = 'prediction_512x512_'
    hparam_str += args.param_alternation + '_' if args.param_alternation else ''
    hparam_str += str(latent_dim) + 'z_'
    hparam_str += str(int(batch_normalization)) + 'BN_'
    hparam_str += str(epochs) + 'ep_'
    hparam_str += datetime_str

    os.makedirs(os.path.join(args.project_path, '4_runs/plots/', hparam_str), exist_ok=True)

    # build the computational graph
    encoder_input = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    z = encoder(encoder_input)[2]

    # build prediction models
    out_full_cl = tf.keras.layers.Dense(1, use_bias=use_bias)(z)
    out_partial_cl = tf.keras.layers.Dense(1, use_bias=use_bias)(z)
    out_loss_cat_4d_prob = tf.keras.layers.Dense(n_loss_cat_4d, use_bias=use_bias)(z)
    out_loss_cat_2d_prob = tf.keras.layers.Dense(n_loss_cat_2d, use_bias=use_bias)(z)
    out_plant_cat_prob = tf.keras.layers.Dense(n_plant_cat, use_bias=use_bias)(z)

    if batch_normalization:
        out_full_cl = tf.keras.layers.BatchNormalization()(out_full_cl)
        out_partial_cl = tf.keras.layers.BatchNormalization()(out_partial_cl)
        out_loss_cat_4d_prob = tf.keras.layers.BatchNormalization()(out_loss_cat_4d_prob)
        out_loss_cat_2d_prob = tf.keras.layers.BatchNormalization()(out_loss_cat_2d_prob)
        out_plant_cat_prob = tf.keras.layers.BatchNormalization()(out_plant_cat_prob)

    out_full_cl = tf.keras.layers.Activation(activation='sigmoid', name='out_full_cl')(out_full_cl)
    out_partial_cl = tf.keras.layers.Activation(activation='sigmoid', name='out_partial_cl')(out_partial_cl)
    out_loss_cat_4d_prob = tf.keras.layers.Activation(activation='softmax', name='out_loss_cat_4d_prob')(out_loss_cat_4d_prob)
    out_loss_cat_2d_prob = tf.keras.layers.Activation(activation='softmax', name='out_loss_cat_2d_prob')(out_loss_cat_2d_prob)
    out_plant_cat_prob = tf.keras.layers.Activation(activation='softmax', name='out_plant_cat_prob')(out_plant_cat_prob)

    # instantiate prediction model
    pred_model = tf.keras.models.Model(
        inputs=encoder_input,
        outputs=[out_full_cl, out_partial_cl, out_loss_cat_4d_prob, out_loss_cat_2d_prob, out_plant_cat_prob],
        name='pred_model')
    rmsprop = tf.keras.optimizers.RMSprop(lr=0.0001)

    pred_model.compile(optimizer=rmsprop,
                       loss=['mse', 'mse', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                       metrics={'out_loss_cat_2d_prob': [f1, 'accuracy'],
                                'out_loss_cat_4d_prob': 'accuracy',
                                'out_plant_cat_prob': 'accuracy'})
    pred_model.summary()
    tf.keras.utils.plot_model(
        pred_model, to_file=os.path.join(args.project_path, '4_runs/plots/', hparam_str, 'pred_model.png'),
        show_shapes=True)

    # add callbacks for:
    # - TensorBoard logger
    # - logging and for saving model checkpoints

    tbCallBack = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.project_path, '4_runs/logging/TBlogs/' + hparam_str),
        histogram_freq=0,  # TODO: fix error when setting histogram_freq > 0
        batch_size=batch_size,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
        embeddings_data=None,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.project_path, '4_runs/logging/checkpoints/pred_model_' + hparam_str + '.hdf5'),
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1,
    )
    callbacks_list = [
        model_checkpoint,
        tbCallBack
    ]

    print('start the training')
    pred_model.fit(
        ds_train,
        steps_per_epoch=steps_per_epoch_train if not args.debug else 3,
        validation_data=ds_val,
        validation_steps=steps_per_epoch_val if not args.debug else 3,
        epochs=epochs if not args.debug else 3,
        callbacks=callbacks_list,
    )
    print('training done')

    # save models and weights
    dir_models = os.path.join(args.project_path, '4_runs/logging/models/', hparam_str)
    os.makedirs(dir_models, exist_ok=True)
    pred_model.save(os.path.join(dir_models, 'pred_model.hdf5'))

    dir_weights = os.path.join(args.project_path, '4_runs/logging/weights/', hparam_str)
    os.makedirs(dir_weights, exist_ok=True)
    pred_model.save_weights(os.path.join(dir_weights, 'pred_model.h5'))
    print('models and weights saved')

    print('evaluate the model')
    scores = pred_model.evaluate(
        x=ds_test,
        steps=steps_per_epoch_test if not args.debug else 3,
    )

    for score, metric in zip(np.round(scores, 3), pred_model.metrics_names):
        print(f'{metric}: {score}')

    print('prediction')
    y_pred_full_cl, y_pred_partial_cl, y_pred_loss_cat_4d_prob, y_pred_loss_cat_2d_prob, y_pred_plant_cat_prob = pred_model.predict(
        x=ds_test,
        steps=steps_per_epoch_test if not args.debug else 3,
        workers=os.cpu_count(),
        use_multiprocessing=True,
    )

    print()

    # get access to elements from tf.data.Dataset object
    # Create an one-shot iterator over the dataset
    sess = tf.Session()
    iterator = ds_test.make_one_shot_iterator()
    # return tuple of X, y (depends on what _parse_function returns)
    _, _y_test = iterator.get_next()
    # get access to arrays through session
    y_test_full_cl = np.array([]).reshape(0, 1)
    y_test_partial_cl = np.array([]).reshape(0, 1)
    y_test_loss_cat_4d_one_hot = np.array([]).reshape(0, n_loss_cat_4d)
    y_test_loss_cat_2d_one_hot = np.array([]).reshape(0, n_loss_cat_2d)
    y_test_plant_cat_one_hot = np.array([]).reshape(0, n_plant_cat)

    for _ in range(steps_per_epoch_test):
        full_cl, partial_cl, loss_cat_4d_one_hot, loss_cat_2d_one_hot, plant_cat_one_hot = sess.run(_y_test)

        full_cl = np.reshape(full_cl, (len(full_cl), 1))
        partial_cl = np.reshape(partial_cl, (len(full_cl), 1))

        y_test_full_cl = np.concatenate((y_test_full_cl, full_cl))
        y_test_partial_cl = np.concatenate((y_test_partial_cl, partial_cl))
        y_test_loss_cat_4d_one_hot = np.concatenate((y_test_loss_cat_4d_one_hot, loss_cat_4d_one_hot))
        y_test_loss_cat_2d_one_hot = np.concatenate((y_test_loss_cat_2d_one_hot, loss_cat_2d_one_hot))
        y_test_plant_cat_one_hot = np.concatenate((y_test_plant_cat_one_hot, plant_cat_one_hot))

    # store arrays to keep them constant over all epochs
    sess.close()

    plot_confusion_matrix(
        np.argmax(y_test_loss_cat_4d_one_hot, axis=1),
        np.argmax(y_pred_loss_cat_4d_prob, axis=1),
        class_names=['full loss', 'partial loss', 'full and partial loss', 'no loss'],
        path=os.path.join(args.project_path, '4_runs/plots/', hparam_str),
        file_name_prefix='loss_cat_4d',
        normalize=False
    )

    plot_confusion_matrix(
        np.argmax(y_test_loss_cat_2d_one_hot, axis=1),
        np.argmax(y_pred_loss_cat_2d_prob, axis=1),
        class_names=['no loss', 'some loss'],
        path=os.path.join(args.project_path, '4_runs/plots/', hparam_str),
        file_name_prefix='loss_cat_2d',
        normalize=False
    )

    # top_5_plants = ['Rehuohra', 'Kaura', 'Mallasohra', 'Kevätvehnä', 'Kevätrypsi']
    top_5_plants = ['Feed Barley', 'Oats', 'Malting Barley', 'Spring Wheat', 'Spring Rapeseed']

    plot_confusion_matrix(
        np.argmax(y_test_plant_cat_one_hot, axis=1),
        np.argmax(y_pred_plant_cat_prob, axis=1),
        class_names=top_5_plants,
        path=os.path.join(args.project_path, '4_runs/plots/', hparam_str),
        file_name_prefix='plant_cat',
        normalize=False
    )

    print()
