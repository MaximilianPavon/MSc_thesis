import numpy as np
import keras
from skimage import io


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, path_to_data, has_cb_and_ext, colour_band=None, file_extension=None, batch_size=32,
                 dim=(512,512), n_channels=13, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path_to_data = path_to_data
        self.has_cb_and_ext = has_cb_and_ext
        self.colour_band = colour_band
        self.file_extension = file_extension
        self.n = len(list_IDs)
        self.step_size = self.n // self.batch_size

        if not self.has_cb_and_ext and (not self.colour_band or not self.file_extension):
            raise Exception('When has_cb_and_ext is set to False, both colour_band and file_extension have to be '
                            'specified')

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        # return X , y
        return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            if self.has_cb_and_ext:
                X[i,] = io.imread(self.path_to_data + ID)
            else:
                X[i,] = io.imread(self.path_to_data + ID + '_' + self.colour_band + self.file_extension)

            # Store class
            y[i] = self.labels[ID]

        return X, y


class TensorBoardWrapper(keras.callbacks.TensorBoard):
    """Sets the self.validation_data property for use with TensorBoard callback."""

    def __init__(self, batch_gen, nb_steps, b_size, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.
        self.nb_steps = nb_steps  # Number of times to call next() on the generator.
        # self.batch_size = b_size

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.

        # adaptation due to the fact that right now pairs of X, X are returned from the data generator and hence
        # the image has to be returned also here twice for self.validation_data

        imgs, tags = None, None
        # for s in range(self.nb_steps):
        s = 0
        for ib, tb in self.batch_gen:
            # ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * self.batch_size,) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * self.batch_size,) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            # tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
            s += 1

        # self.validation_data = [imgs, tags, np.ones(imgs.shape[0])]
        self.validation_data = [imgs, imgs, np.ones(imgs.shape[0])]

        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)
