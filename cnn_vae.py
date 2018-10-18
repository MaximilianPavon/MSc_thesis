import pandas as pd
import numpy as np
from PIL import Image
import os.path
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

tfd = tf.contrib.distributions


def preprocess_df(_path_to_csv):
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
    print(_df.shape[0] - len(np.unique(_df['field parcel'])), 'duplicate entries')
    fieldparcel = _df['field parcel']
    _df = _df[fieldparcel.duplicated() == False]
    print(_df.shape[0] - len(np.unique(_df['field parcel'])), 'duplicate entries')

    print('total number of fields: ', _df.shape[0])

    # select only those rows, where both full and partial crop loss are present
    # print('only ', len(_df['full crop loss'].dropna()), '(full) resp. ', len(_df['partial crop loss'].dropna()),
    #       '(partial) of a total ', len(_df), 'have a crop loss information')
    # _df = _df.dropna(subset=['full crop loss', 'partial crop loss'])
    # print('total number of fields where both full and partial crop loss are present : ', len(_df))

    print('create new column: relative crop loss = crop loss / Property area')
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
    print('maximum number for', plant_max, 'with', num, 'entries')
    _df = _df[plants == plant_max]

    col_list = ['field parcel', 'full crop loss scaled', 'partial crop loss scaled']

    print('trim data frame to:', col_list)
    return _df[col_list]


def return_images_and_croploss(_df):
    files = []
    _full_cl = []
    _partial_cl = []

    t = 5

    path_to_data = 'data/'
    layer = 'NDVI'
    f_extension = '.png'

    for index, row in _df.iterrows():

        if len(files) + 1 > t:
            break

        file = path_to_data + str(row['field parcel']) + '_' + layer + f_extension
        if os.path.isfile(file):
            files.append(file)
            _full_cl.append(row['full crop loss scaled'])
            _partial_cl.append(row['partial crop loss scaled'])

    _full_cl = np.array(_full_cl)
    _partial_cl = np.array(_partial_cl)

    _x = np.array([np.array(Image.open(fname)) for fname in files])

    return _x, _full_cl, _partial_cl


def unpool_2d(pool, ind, stride=[1, 2, 2, 1], scope='unpool_2d'):
    """Adds a 2D unpooling op.
  https://arxiv.org/abs/1505.04366

  Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
           scope:       ??
       Return:
           unpool:    unpooling tensor
  """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2],
                            set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret


def next_batch(num, data, labels):
    """
    Return a total of `num` random samples and labels.
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


if __name__ == '__main__':
    path_to_csv = 'MAVI2/2015/rap_2015.csv'

    df = preprocess_df(path_to_csv)

    x, y_full_cl, y_partial_cl = return_images_and_croploss(df)

    # ======================
    # Graph Definition: Define the Encoder
    # ======================

    # Input Layer
    # Input 4-D tensor: [batch_size, width, height, channels]
    # images are 512x512 pixels, and have 3 color channel
    input_image = tf.placeholder(tf.float32, [None, 512, 512, 3])

    # Convolutional Layer #1
    # Computes 16 features using a 10x10 filter and stride of 2 with ReLU activation.
    # Input Tensor Shape: [batch_size, 512, 512, 3]
    # Output Tensor Shape: [batch_size, 252, 252, 16]
    conv1 = tf.layers.conv2d(
        inputs=input_image,
        filters=16,
        kernel_size=[10, 10],
        strides=2,
        padding="valid",
        activation=tf.nn.relu,
        name='conv1')

    # Pooling Layer #1
    # First max pooling layer with a 3x3 filter and stride of 3
    # Input Tensor Shape: [batch_size, 252, 252, 16]
    # Output Tensor Shape: [batch_size, 84, 84, 16]
    # pool1_l = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)
    pool1, pool1_am = tf.nn.max_pool_with_argmax(
        input=conv1,
        ksize=[1, 3, 3, 1],
        strides=[1, 3, 3, 1],
        padding='VALID',
        name='pool1'
    )

    # Convolutional Layer #2
    # Computes 16 features using a 10x10 filter and stride of 2.
    # Input Tensor Shape: [batch_size, 84, 84, 16]
    # Output Tensor Shape: [batch_size, 38, 38, 16]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=16,
        kernel_size=[10, 10],
        strides=2,
        padding="valid",
        activation=tf.nn.relu,
        name='conv2')

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 38, 38, 16]
    # Output Tensor Shape: [batch_size, 19, 19, 16]
    # pool2_l = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2, pool2_am = tf.nn.max_pool_with_argmax(
        input=conv2,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pool2'
    )

    # Convolutional Layer #3
    # Computes 16 features using a 5x5 filter and stride of 2.
    # Input Tensor Shape: [batch_size, 19, 19, 16]
    # Output Tensor Shape: [batch_size, 8, 8, 16]
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=16,
        kernel_size=[5, 5],
        strides=2,
        padding="valid",
        activation=tf.nn.relu,
        name='conv3')

    # Pooling Layer #3
    # Third max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 8, 8, 16]
    # Output Tensor Shape: [batch_size, 4, 4, 16]
    # pool3_l = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    pool3, pool3_am = tf.nn.max_pool_with_argmax(
        input=conv3,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='VALID',
        name='pool3'
    )

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 4, 4, 16]
    # Output Tensor Shape: [batch_size, 4 * 4 * 16]
    pool3_flat = tf.reshape(pool3, [-1, 4 * 4 * 16])

    # learn mean and log(standard deviation) with densely connected layers
    # Input Tensor Shape: [batch_size, 4 * 4 * 16]
    # Output Tensor Shape: [batch_size, 2]
    z_mean = tf.layers.dense(pool3_flat, 2, activation=tf.nn.relu, name='dense')
    z_log_var = tf.layers.dense(pool3_flat, 2, tf.nn.softplus)

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # then z = z_mean + z_std * eps
    eps = tf.random_normal(tf.shape(z_log_var), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
    z = z_mean + tf.exp(z_log_var / 2) * eps

    # ======================
    # Graph Definition: Define the Decoder
    # ======================

    # reverse fully connected layer
    # Input Tensor Shape: [batch_size, 2]
    # Output Tensor Shape: [batch_size, 4 * 4 * 16]
    z_fc_flat = tf.layers.dense(z, 4 * 4 * 16, activation=tf.nn.relu, name='undense')

    # Unflatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 4 * 4 * 16]
    # Output Tensor Shape: [batch_size, 4, 4, 16]
    z_fc_unflat = tf.reshape(z_fc_flat, [-1, 4, 4, 16])

    # Unpooling Layer #3
    # Unpooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 4, 4, 16]
    # Output Tensor Shape: [batch_size, 8, 8, 16]
    unpool3 = unpool_2d(pool=z_fc_unflat, ind=pool3_am)

    # Deconvolutional Layer #3
    # Computes 16 features using a 5x5 filter and stride of 2.
    # Input Tensor Shape: [batch_size, 8, 8, 16]
    # Output Tensor Shape: [batch_size, 19, 19, 16]
    filter_deconv3 = tf.get_variable(name="filter_deconv3", shape=[5, 5, 16, 16],
                                     initializer=tf.initializers.random_normal)
    deconv3 = tf.nn.relu(
        tf.nn.conv2d_transpose(
            value=unpool3,
            filter=filter_deconv3,
            output_shape=[-1, 19, 19, 16],
            strides=[1, 2, 2, 1],
            padding='VALID',
            name='deconv3'
        )
    )

    # Unpooling Layer #2
    # Unpooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 19, 19, 16]
    # Output Tensor Shape: [batch_size, 38, 38, 16]
    unpool2 = unpool_2d(pool=deconv3, ind=pool2_am)

    # Deconvolutional Layer #2
    # Computes 16 features using a 10x10 filter and stride of 2.
    # Input Tensor Shape: [batch_size, 38, 38, 16]
    # Output Tensor Shape: [batch_size, 84, 84, 16]
    filter_deconv2 = tf.get_variable(name="filter_deconv2", shape=[10, 10, 16, 16],
                                     initializer=tf.initializers.random_normal)
    deconv2 = tf.nn.relu(
        tf.nn.conv2d_transpose(
            value=unpool2,
            filter=filter_deconv2,
            output_shape=[-1, 84, 84, 16],
            strides=[1, 2, 2, 1],
            padding='VALID',
            name='deconv2'
        )
    )

    # Unpooling Layer #1
    # Unpooling layer layer with a 3x3 filter and stride of 3
    # Input Tensor Shape: [batch_size, 84, 84, 16]
    # Output Tensor Shape: [batch_size, 252, 252, 16]
    unpool1 = unpool_2d(pool=deconv2, ind=pool1_am, stride=[1, 3, 3, 1])

    # Deconvolutional Layer #1
    # Computes 16 features using a 10x10 filter and stride of 2 with ReLU activation.
    # Input Tensor Shape: [batch_size, 252, 252, 16]
    # Output Tensor Shape: [batch_size, 512, 512, 3]
    filter_deconv1 = tf.get_variable(name="filter_deconv1", shape=[10, 10, 3, 16],
                                     initializer=tf.initializers.random_normal)
    recon_image = tf.nn.relu(
        tf.nn.conv2d_transpose(
            value=unpool1,
            filter=filter_deconv1,
            output_shape=[-1, 512, 512, 3],
            strides=[1, 2, 2, 1],
            padding='VALID',
            name='deconv1'
        )
    )

    # The loss is composed of two terms:
    # 1.) The reconstruction loss (the negative log probability
    #     of the input under the reconstructed Bernoulli distribution
    #     induced by the decoder in the data space).
    #     This can be interpreted as the number of "nats" required
    #     for reconstructing the input when the activation in latent
    #     is given.
    # Adding 1e-10 to avoid evaluation of log(0.0)
    reconstr_loss = -tf.reduce_sum(
        input_image * tf.log(1e-10 + recon_image) + (1 - input_image) * tf.log(1e-10 + 1 - recon_image),
        1
    )

    # 2.) The latent loss, which is defined as the Kullback Leibler divergence
    #     between the distribution in latent space induced by the encoder on
    #     the data and some prior. This acts as a kind of regularizer.
    #     This can be interpreted as the number of "nats" required
    #     for transmitting the the latent space distribution given
    #     the prior.
    KL = -0.5 * tf.reduce_sum(1 + z_log_var
                              - tf.square(z_mean)
                              - tf.exp(z_log_var), 1)

    # Total loss
    loss = tf.reduce_mean(reconstr_loss + KL)  # average over batch

    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 5

    # Use ADAM optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    # Seed the random number generator for reproducible batches
    np.random.seed(0)

    # Print list of variables
    print("")
    print("Variables")
    print("---------")
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    num_params = 0
    for v in variables:
        num_params += np.prod(v.get_shape().as_list())
        print(v.name, v.get_shape())
    print("=> Total number of parameters =", num_params)

    # TF session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # save graph for visualisation purposes
    tf.train.write_graph(sess.graph_def, 'my-model/', 'graph.pb', as_text=False)
    tf.train.write_graph(sess.graph_def, 'my-model/', 'graph.prototxt', as_text=True)

    ##### Debugging
    # conv1_, conv2_, conv3_, pool1_, pool2_, pool3_, pool3_flat_, z_mean_, z_log_var_ = sess.run(
    #     [conv1, conv2, conv3, pool1, pool2, pool3, pool3_flat, z_mean, z_log_var],
    #     feed_dict={input_image: x})

    # z_fc_unflat_ = sess.run(
    #     [z_fc_unflat],
    #     feed_dict={input_image: x})

    unpool3_ = sess.run(
        [unpool3],
        feed_dict={input_image: x})

    print('end of debuggin')


    # Minimize the loss function
    num_batches_per_epoch = len(x) // batch_size
    for epoch in range(num_epochs):
        current_loss = 0
        for _ in range(num_batches_per_epoch):
            batch_x, y_full_cl = next_batch(batch_size, x, y_full_cl)
            loss_val = sess.run(loss, {input_image: batch_x})
            current_loss += loss_val

        if (epoch + 1) % 5 == 0:
            print("After {} epochs, loss = {}"
                  .format(epoch + 1, current_loss / num_batches_per_epoch))

    print('end')
