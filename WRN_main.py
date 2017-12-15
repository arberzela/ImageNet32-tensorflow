"""Contains definitions for the preactivation form of Residual Networks.
Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys, os, time, pickle
from argparse import ArgumentParser
import numpy as np
import logging

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-4

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

num_of_train_images = 1281167

class Logger(object):
    def __init__(self, k, lr, run):
        self.log = logging.getLogger('Log Message')
        self.stat = logging.getLogger('Log Stat')
        self.loss = logging.getLogger('Log Loss')
        
        self.log.setLevel(logging.DEBUG)
        self.stat.setLevel(logging.DEBUG)
        self.loss.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(message)s')

        fh1 = logging.FileHandler('log_{}_{}_{}.txt'.format(k, lr, run))
        fh1.setFormatter(formatter)
        self.log.addHandler(fh1)

        fh2 = logging.FileHandler("stat_{}_{}_{}.txt".format(k, lr, run))
        fh2.setFormatter(formatter)
        self.stat.addHandler(fh2)

        fh3 = logging.FileHandler("statloss_{}_{}_{}.txt".format(k, lr, run))
        fh3.setFormatter(formatter)
        self.loss.addHandler(fh3)
        
    def log_message(self, message):
        self.log.info(message)
        
    def log_stat(self, message):
        self.stat.info(message)
        
    def log_loss(self, message):
        self.loss.info(message)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


# Mean image can be extracted from any training data file
def load_validation_data(data_folder, mean_image, img_size=32):
    test_file = os.path.join(data_folder, 'val_data')

    d = unpickle(test_file)
    x = d['data']
    y = d['labels']
    x = x / np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = np.array([i-1 for i in y])

    # Remove mean (computed from training data) from images
    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    return dict(
        X_test=x.astype('float32'),
        Y_test=y.astype('int32'))


def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(
        X_train=X_train.astype('float32'),
        Y_train=Y_train.astype('int32'),
        mean=mean_image)

def batch_norm_relu(inputs, is_training, data_format):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  inputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=is_training, fused=True)
  inputs = tf.nn.relu(inputs)
  return inputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding='SAME', use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(scale=2.0, distribution='normal'),
      data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
                   dropoutrate, data_format):
  """Standard building block for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = tf.layers.dropout(inputs=inputs, rate=dropoutrate, training=is_training)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)

  return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
                     strides, data_format):
  """Bottleneck block variant for residual networks with BN before convolutions.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block.
  """
  shortcut = inputs
  inputs = batch_norm_relu(inputs, is_training, data_format)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm_relu(inputs, is_training, data_format)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_group(inputs, filters, block_fn, blocks, strides, dropoutrate, is_training, name,
                data_format):
  """Creates one layer of blocks for the ResNet model.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    The output tensor of the block layer.
  """
  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = 4 * filters if block_fn is bottleneck_block else filters

  def projection_shortcut(inputs):
    return tf.layers.conv2d(
      inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
      padding='SAME', use_bias=False,
      kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
      data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                    dropoutrate, data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, None, 1,
                      dropoutrate, data_format)

  return tf.identity(inputs, name)


# ##################### Build the neural network model #######################


def ImgNet_downsampled_WRN_v2_generator(depth=16, k=2, num_classes=1000, data_format=None, dropoutrate=0, img_size=32):
  """Generator for downsampled ImageNet WRN v2 models.
  Args:
    depth: A single integer for the size of the WRN model.
    k: The filter multiplicative factor which determines the width of the network.
    num_classes: The number of possible classes for image classification.
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
  Returns:
    The model function that takes in `inputs` and `is_training` and
    returns the output tensor of the WRN model.
  Raises:
    ValueError: If `depth` is invalid.
  """
  if depth % 6 != 4:
    raise ValueError('depth must be 6n + 4:', depth)

  num_blocks = (depth - 4) // 6

  if data_format is None:
    data_format = (
        'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

  def model(inputs, is_training):
    """Constructs the ResNet model given the inputs."""

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=16, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    num_filters = int(16*k)
    inputs = block_group(
        inputs=inputs, filters=num_filters, block_fn=building_block, blocks=num_blocks,
        strides=1, dropoutrate=dropoutrate, is_training=is_training, name='block_layer1',
        data_format=data_format)

    if img_size >= 16:
        num_filters = int(32*k)
        inputs = block_group(
            inputs=inputs, filters=num_filters, block_fn=building_block, blocks=num_blocks,
            strides=2, dropoutrate=dropoutrate, is_training=is_training, name='block_layer2',
            data_format=data_format)

    if img_size >= 32:
        num_filters = int(64*k)
        inputs = block_group(
            inputs=inputs, filters=num_filters, block_fn=building_block, blocks=num_blocks,
            strides=2, dropoutrate=dropoutrate, is_training=is_training, name='block_layer3',
            data_format=data_format)

    if img_size >= 64:
        num_filters = int(128*k)
        inputs = block_group(
            inputs=inputs, filters=num_filters, block_fn=building_block, blocks=num_blocks,
            strides=2, dropoutrate=dropoutrate, is_training=is_training, name='block_layer4',
            data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=8, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs, [-1, num_filters])
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    inputs = tf.identity(inputs, 'final_dense')
    return inputs

  return model


# ############################# Batch iterator ###############################


def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False, img_size=32):

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping
            padded = np.pad(inputs[excerpt], ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0, high=8, size=(batchsize, 2))
            for r in range(batchsize):
                random_cropped[r, :, :, :] = \
                    padded[r, :, crops[r, 0]:(crops[r, 0]+img_size), crops[r, 1]:(crops[r, 1]+img_size)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]


# ############################## Main program ################################

def _main(data_folder, depth=16, irun=1, k=2, num_epochs=40, cont=False, E1=10, E2=20, E3=30, lr=0.1, lr_fac=0.1,
         reg_fac=0.0005, dropoutrate=0, img_size=32):

    num_classes = 1000
    logger = Logger(k, lr, irun)

    # Load the dataset
    logger.log_message("Loading data...")

    # Load first batch so we can extract mean image needed to load validation data
    data = load_databatch(data_folder, 1, img_size=img_size)
    mean_image = data['mean']
    del data

    # Load test data
    test_data = load_validation_data(data_folder, mean_image=mean_image, img_size=img_size)
    X_test = test_data['X_test']
    Y_test = test_data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = tf.placeholder(tf.float32, shape=(None, 3, img_size, img_size), name='inputs')
    target_var = tf.placeholder(tf.int32, shape=(None), name='targets')

    # True if training, False if testing
    mode = tf.placeholder(shape=(), name= 'mode', dtype=tf.bool)

    # Create neural network model
    logger.log_message("Building model and compiling functions...")
    network = ImgNet_downsampled_WRN_v2_generator(depth=depth, k=k, num_classes=num_classes, data_format='channels_first', dropoutrate=dropoutrate, img_size=img_size)
    #logger.log_message("Number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
    #print("Number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
    print('Img Size %d' % img_size)
    print('K %d' % k)
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    logits = network(input_var, mode)

    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    onehot_labels = tf.one_hot(indices=tf.cast(target_var, tf.int32), depth=num_classes)
    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=onehot_labels)

    # Create a tensor named cross_entropy for logging purposes.
    #tf.identity(cross_entropy, name='cross_entropy')
    #tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy + reg_fac * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    
    # Create update expressions for training
    # Stochastic Gradient Descent (SGD) with momentum
    sh_lr = tf.placeholder_with_default(input=lr, shape=(), name='learning_rate')
    optimizer = tf.train.MomentumOptimizer(learning_rate=sh_lr, momentum=0.9)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)

    accuracy = tf.metrics.accuracy(
        tf.argmax(onehot_labels, axis=1), predictions['classes'])

    accuracy5 = tf.nn.in_top_k(predictions['probabilities'], tf.argmax(onehot_labels, axis=1), 5)
    accuracy5 = tf.cast(accuracy5, tf.float32)
    accuracy5 = tf.reduce_mean(accuracy5)

    #metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    #tf.identity(accuracy[1], name='train_accuracy')
    #tf.summary.scalar('train_accuracy', accuracy[1])

    start_time0 = time.time()

    batchsize = 128
    start_epoch = 0
    lr = lr

    # Training #####################################################################
    logger.log_message("Starting training...")

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # We iterate over epochs:
        for epoch in range(start_epoch, num_epochs):
            # In each epoch, we do a full pass over the training data:
            start_time = time.time()

            for idatabatch in range(1, 11):
                start_time_tmp = time.time()
                data = load_databatch(data_folder, idatabatch, img_size=img_size)
                print('Data loading took %f' % (time.time() - start_time_tmp))
                X_train = data['X_train']
                Y_train = data['Y_train']

                train_err = 0
                train_batches = 0

                for batch in iterate_minibatches(X_train, Y_train, batchsize, shuffle=True, augment=True, img_size=img_size):
                    inputs, targets = batch
                    train_loss, _ = sess.run([loss, train_op], feed_dict={input_var: inputs, target_var: targets, mode: True, sh_lr: lr})
                    train_err += train_loss
                    train_batches += 1

                logger.log_loss("{}\t{:.15g}\t{:.15g}\t{:.15g}\n".format(epoch, sess.run(sh_lr),
                    time.time() - start_time0, train_err / train_batches))

                logger.log_message("idatabatch#{} took {:.3f}s".format(idatabatch, time.time() - start_time))
                del data, X_train, Y_train

            print('Train Data pass took: %f' % (time.time() - start_time))
        
            # And a full pass over the validation data:
            val_err = 0
            val_acc_1 = 0
            val_acc_5 = 0
            val_batches = 0
            for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False, img_size=img_size):
                inputs, targets = batch
                val_loss, acc_1, acc_5 = sess.run([cross_entropy, accuracy, accuracy5], feed_dict={input_var: inputs, target_var: targets, mode: False, sh_lr: lr})
                val_err += val_loss
                val_acc_1 += acc_1[1]
                val_acc_5 += acc_5
                val_batches += 1

            print('Epoch took: %f' % (time.time() - start_time))
            # Then we print the results for this epoch:
            logger.log_message("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            logger.log_message("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            logger.log_message("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            logger.log_message("  validation accuracy_1:\t\t{:.2f} %".format(val_acc_1 / val_batches * 100))
            logger.log_message("  validation accuracy_5:\t\t{:.2f} %".format(val_acc_5 / val_batches * 100))

            # Print some statistics
            logger.log_stat("{}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\n"
                            .format(epoch, sess.run(sh_lr), time.time() - start_time0,
                                    train_err / train_batches, val_err / val_batches,
                                    val_acc_1 / val_batches * 100))

            # Adjust learning rate
            if (epoch+1) == E1 or (epoch+1) == E2 or (epoch+1) == E3:
                lr = lr * lr_fac
                logger.log_message("New LR: "+str(lr))

        # Calculate validation error of model:
        test_err = 0
        test_acc_1 = 0
        test_acc_5 = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
            inputs, targets = batch
            test_loss, acc_1, acc_5 = sess.run([cross_entropy, accuracy, accuracy5], feed_dict={input_var: inputs, target_var: targets, mode: False, sh_lr: lr})
            test_err += test_loss
            test_acc_1 += acc_1[1]
            test_acc_5 += acc_5
            test_batches += 1
        logger.log_message("Final results:")
        logger.log_message("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        logger.log_message("  test accuracy 1:\t\t{:.2f} %".format(test_acc_1 / test_batches * 100))
        logger.log_message("  test accuracy 5:\t\t{:.2f} %".format(test_acc_5 / test_batches * 100))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-s', '--img_size', help="Size of images, represented as string '32x32' or '64x64'",
                        default=32, type=int)
    parser.add_argument('-lr', '--learning_rate', help="Starting Learning Rate, "
                                                       "decreased by the factor of 5 every 10 epochs",
                        default=0.01, type=float)
    parser.add_argument('-k', '--network_width', help="Network width hyper-parameter. Number of filters in each layer "
                                                      "is multiplied by this factor", default=1.0, type=float)
    parser.add_argument('-r', '--run', help="Number used to index output files, helpful when multiple runs required",
                        default=1, type=int)
    parser.add_argument('-c', '--cont', help="Read last saved model and continue training from that point",
                        default=False, type=bool)
    parser.add_argument('-df', '--data_folder', help="Path to the folder containing training and validation data",
                        required=True)
    parser.add_argument('-d', '--decay', help="L2 decay", default=0.0005, type=float)
    args = parser.parse_args()

    return args.img_size, args.learning_rate, args.network_width, args.run, args.cont, args.data_folder, args.decay


def main(argv=None):
    img_size, lr, k, run, cont, data_folder, reg_fac = parse_arguments()

    lr_fac = 0.2
    num_epochs = 40
    E1 = 10
    E2 = 20
    E3 = 30
    Estart = 10000
    depth = 28
    dropout = 0

    _main(data_folder, depth, run, k, num_epochs, cont, E1, E2, E3, lr, lr_fac, reg_fac, dropout, img_size)

if __name__ == '__main__':
    tf.app.run()
