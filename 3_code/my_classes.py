import time
from threading import Thread
from my_functions import get_device_util
import tensorflow as tf
import matplotlib.pyplot as plt
import os, glob
import imageio
from skimage import exposure


class Monitor(Thread):
    def __init__(self, delay, gpu_device_ID):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.gpu_device_ID = gpu_device_ID
        self.gpu_util = -100
        self.start()

    def run(self):
        while not self.stopped:
            self.gpu_util = get_device_util(self.gpu_device_ID)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


class MyCallbackDecoder(tf.keras.callbacks.Callback):
    def __init__(self, decoder, log_dir, num_examples_to_generate=16, log_freq=10):
        super(MyCallbackDecoder, self).__init__()
        self.decoder = decoder
        self.log_dir = log_dir
        self.num_examples_to_generate = num_examples_to_generate
        self.latent_dim = self.decoder.input[0].shape.dims[0].value
        self.log_freq = log_freq
        os.makedirs(self.log_dir, exist_ok=True)

        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        self.random_vector_for_generation = tf.random_normal(shape=[self.num_examples_to_generate, self.latent_dim])

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.log_freq == 0:
            predictions = self.decoder.predict(self.random_vector_for_generation, steps=1)
            fig = plt.figure(figsize=(4, 4))

            for i in range(predictions.shape[0]):
                plt.subplot(4, 4, i + 1)
                im = predictions[i][ :, :, [3, 2, 1]]
                im = exposure.rescale_intensity(im)
                plt.imshow(im)
                plt.axis('off')

            # tight_layout minimizes the overlap between 2 sub-plots
            plt.savefig(
                os.path.join(self.log_dir, 'image_at_epoch_{:04d}.png'.format(epoch)),
                dpi=100, bbox_inches='tight')
            # plt.show()
            plt.clf()
            pass
        else:
            pass

    def on_train_end(self, logs=None):
        # merge individual images to one .gif
        with imageio.get_writer(os.path.join(self.log_dir, 'decoded.gif'), mode='I') as writer:
            filenames = glob.glob(os.path.join(self.log_dir, 'image*.png'))
            filenames = sorted(filenames)
            last = -1
            for i, filename in enumerate(filenames):
                frame = 2 * (i ** 0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

        # os.system('rm image_at_epoch_0*')  # optional clean up - not adapted yet
        pass


class MyCallbackCompOrigDecoded(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, dataset, num_examples=10, log_freq=10):
        super(MyCallbackCompOrigDecoded, self).__init__()
        self.log_dir = log_dir
        self.dataset = dataset
        self.num_examples = num_examples
        self.log_freq = log_freq
        os.makedirs(self.log_dir, exist_ok=True)

        # get access to elements from tf.data.Dataset object
        sess = tf.Session()
        # Create an iterator over the dataset and initialize the iterator
        iterator = self.dataset.take(self.num_examples).make_initializable_iterator()
        sess.run(iterator.initializer)
        # return tuple of images, images (depends on what _parse_function returns)
        im1, im2 = iterator.get_next()
        # get access to arrays
        im1_arr = sess.run(im1)
        if im1_arr.shape[0] > self.num_examples:
            im1_arr = im1_arr[0:self.num_examples]
        # store arrays to keep them constant over all epochs
        self.orig_images = im1_arr

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.log_freq == 0:
            predictions = self.model.predict(self.orig_images, steps=1)

            fig = plt.figure(figsize=(self.num_examples, 2))

            for i in range(self.num_examples):
                # first row (1..... num_examples) holds the original images
                plt.subplot(2, self.num_examples, i + 1)
                orig_im = self.orig_images[i][ :, :, [3, 2, 1]]
                orig_im = exposure.rescale_intensity(orig_im)
                plt.imshow(orig_im)
                plt.axis('off')

                # second row (1 + num_examples ...... 2 * num_examples) shows the decoded images
                plt.subplot(2, self.num_examples,  i + self.num_examples + 1)
                decoded_im = predictions[i][ :, :, [3, 2, 1]]
                decoded_im = exposure.rescale_intensity(decoded_im)
                plt.imshow(decoded_im)
                plt.axis('off')

            # tight_layout minimizes the overlap between 2 sub-plots
            plt.savefig(
                os.path.join(self.log_dir, 'image_at_epoch_{:04d}.png'.format(epoch)),
                dpi=100, bbox_inches='tight')
            # plt.show()
            plt.clf()
            pass
        else:
            pass

    def on_train_end(self, logs=None):
        # merge individual images to one .gif
        with imageio.get_writer(os.path.join(self.log_dir, 'comparison_orig_decoded.gif'), mode='I') as writer:
            filenames = glob.glob(os.path.join(self.log_dir, 'image*.png'))
            filenames = sorted(filenames)
            last = -1
            for i, filename in enumerate(filenames):
                frame = 2 * (i ** 0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

        # os.system('rm image_at_epoch_0*')  # optional clean up - not adapted yet
        pass

