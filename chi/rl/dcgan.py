# Doesn't work yet
from time import sleep

import os
import PIL
import scipy.misc
import math
import chi
import tensortools as tt
from tensortools import Function
import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib import layers
from chi.rl.util import pp, to_json, show_all_variables
import argparse

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN:
    """
    An implementation of
        Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
        https://arxiv.org/abs/1511.06434, adapted from https://github.com/carpedm20/DCGAN-tensorflow
    """
    def __init__(self, generator:tt.Model, discriminator:tt.Model, sampler:tt.Model, input_height, input_width,
            output_height=64, output_width=64,
            batch_size=64, y_dim=None,
            z_dim=100, gfc_dim=1024, dfc_dim=1024,
            c_dim=3, dataset_name='default', input_fname_pattern='*.jpg',
            checkpoint_dir=None, sample_dir=None):

        self.batch_size = batch_size
        self.output_height = output_height
        self.output_width = output_width
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.discriminator = discriminator
        self.generator = generator
        self.sampler = sampler

        # batch normalization
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name == 'mnist':
            self.data_X, self.data_y = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
        else:
            self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
            imreadImg = imread(self.data[0]);
            if len(imreadImg.shape) >= 3: # grayscale image?
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G                                  = self.generator(self.z, self.y)
        self.D, self.D_logits       = self.discriminator(inputs, self.y, reuse=False)
        self.sampler                        = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
                [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

        if config.dataset == 'mnist':
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [
                    get_image(sample_file,
                                        input_height=self.input_height,
                                        input_width=self.input_width,
                                        resize_height=self.output_height,
                                        resize_width=self.output_width,
                                        crop=self.crop,
                                        grayscale=self.grayscale) for sample_file in sample_files]
            if (self.grayscale):
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
            else:
                self.data = glob(os.path.join(
                    "./data", config.dataset, self.input_fname_pattern))
                batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                else:
                    batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch = [
                            get_image(batch_file,
                                                input_height=self.input_height,
                                                input_width=self.input_width,
                                                resize_height=self.output_height,
                                                resize_width=self.output_width,
                                                crop=self.crop,
                                                grayscale=self.grayscale) for batch_file in batch_files]
                    if self.grayscale:
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                if config.dataset == 'mnist':
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={
                            self.inputs: batch_images,
                            self.z: batch_z,
                            self.y:batch_labels,
                        })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={
                            self.z: batch_z,
                            self.y:batch_labels,
                        })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({
                            self.z: batch_z,
                            self.y:batch_labels
                    })
                    errD_real = self.d_loss_real.eval({
                            self.inputs: batch_images,
                            self.y:batch_labels
                    })
                    errG = self.g_loss.eval({
                            self.z: batch_z,
                            self.y: batch_labels
                    })
                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.inputs: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z })
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
                    errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
                    errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                    self.z: sample_z,
                                    self.inputs: sample_inputs,
                                    self.y:sample_labels,
                            }
                        )
                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    else:
                        try:
                            samples, d_loss, g_loss = self.sess.run(
                                [self.sampler, self.d_loss, self.g_loss],
                                feed_dict={
                                        self.z: sample_z,
                                        self.inputs: sample_inputs,
                                },
                            )
                            save_images(samples, image_manifold_size(samples.shape[0]),
                                        './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                        except:
                            print("one pic error!...")

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)
        pass

    def load_mnist(self, config):
        data_dir = os.path.join("./data", self.dataset_name)

        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0

        return X/255.,y_vec

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
                self.dataset_name, self.batch_size,
                self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



def test_dcgan():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=25,
            help="Epoch to train [25]")
    parser.add_argument("--learning_rate", type=float, default=0.0002,
            help="Learning rate for adam [0.0002]")
    parser.add_argument("--beta1", type=float, default=0.5,
            help="Momentum term of adam [0.5]")
    parser.add_argument("--train_size", default=np.inf,
            help="The size of train images [np.inf]")
    parser.add_argument("--batch_size", default=64, type=int,
            help="The size of batch images [64]")
    parser.add_argument("--input_height", default=108, type=int,
            help="The size of image to use (will be center cropped) [108]")
    parser.add_argument("--input_width", type=int, default=None,
            help="The size of image to use (will be center cropped).
            If None, same value as input_height [None]")
    parser.add_argument("--output_height", type=int, default=64,
            help="The size of the output images to produce [64]")
    parser.add_argument("--output_width", type=int, default=None,
            help="The size of the output images to produce.
            If None, same value as output_height [None]")
    parser.add_argument("--dataset", type=str, default="celebA",
            help="The name of the dataset [celebA]")
    parser.add_argument("--input_fname_pattern", type=str, default="*.jpg",
            help="Glob pattern of filename of input images [*.jpg]")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
            help="Name of the directory in which to save the checkpoints [checkpoints]")
    parser.add_argument("--sample_dir", type=str, default="samples",
            help="Name of the directory in which to save the images samples [samples]")
    parser.add_argument("--train", type=bool, default=False,
            help="True for training, False for testing [False]")
    parser.add_argument("--crop", type=bool, default=False,
            help="True for cropping [False]")
    parser.add_argument("--visualize", type=bool, default=False,
            help="True for visualizing [False]")
    args = parser.parse_args()

    input_height = args.input_height
    input_width = args.input_width if args.input_width is not None else input_height
    output_height = args.output_height
    output_width = args.output_width if args.output_width is not None else output_height
    batch_size = args.batch_size
    sample_num = args.batch_size
    if args.dataset == 'mnist':
        y_dim = 10
    else:
        y_dim = None
    dataset_name = args.dataset
    input_fname_pattern = args.input_fname_pattern
    crop = args.crop
    checkpoint_dir = args.checkpoint_dir
    sample_dir = args.sample_dir

    gf_dim = 64
    df_dim = 64
    gfc_dim = 1024
    dfc_dim = 1024
    c_dim = 3
    z_dim = 100

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    show_all_variables()


    @tt.model(optimizer=tf.train.AdamOptimizer(0.0002, beta1=0.5))
    def generator(z, y=None):
        with tf.variable_scope("generator") as scope:
            if not y_dim:
                s_h, s_w = output_height, output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                z_, h0_w, h0_b = linear(
                        z, gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

                h0 = tf.reshape(
                        z_, [-1, s_h16, s_w16, gf_dim * 8])
                h0 = tf.nn.relu(g_bn0(h0))

                h1, h1_w, h1_b = deconv2d(
                        h0, [batch_size, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(g_bn1(h1))

                h2, h2_w, h2_b = deconv2d(
                        h1, [batch_size, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(g_bn2(h2))

                h3, h3_w, h3_b = deconv2d(
                        h2, [batch_size, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(g_bn3(h3))

                h4, h4_w, h4_b = deconv2d(
                        h3, [batch_size, s_h, s_w, c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = output_height, output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                        g_bn0(linear(z, gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(g_bn1(
                        linear(h0, gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(g_bn2(deconv2d(h1,
                        [batch_size, s_h2, s_w2, gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                        deconv2d(h2, [batch_size, s_h, s_w, c_dim], name='g_h3'))


    @tt.model(optimizer=tf.train.AdamOptimizer(0.0002, beta1=0.5))
    def discriminator(image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not y_dim:
                h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))
                h1 = lrelu(d_bn1(conv2d(h0, df_dim*2, name='d_h1_conv')))
                h2 = lrelu(d_bn2(conv2d(h1, df_dim*4, name='d_h2_conv')))
                h3 = lrelu(d_bn3(conv2d(h2, df_dim*8, name='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h4_lin')

                return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, c_dim + y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(d_bn1(conv2d(h0, df_dim + y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [batch_size, -1])
                h1 = concat([h1, y], 1)

                h2 = lrelu(d_bn2(linear(h1, dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    @tt.model
    def sampler(z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not y_dim:
                s_h, s_w = output_height, output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                        linear(z, gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                        [-1, s_h16, s_w16, gf_dim * 8])
                h0 = tf.nn.relu(g_bn0(h0, train=False))

                h1 = deconv2d(h0, [batch_size, s_h8, s_w8, gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(g_bn1(h1, train=False))

                h2 = deconv2d(h1, [batch_size, s_h4, s_w4, gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(g_bn2(h2, train=False))

                h3 = deconv2d(h2, [batch_size, s_h2, s_w2, gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(g_bn3(h3, train=False))

                h4 = deconv2d(h3, [batch_size, s_h, s_w, c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = output_height, output_width
                s_h2, s_h4 = int(s_h/2), int(s_h/4)
                s_w2, s_w4 = int(s_w/2), int(s_w/4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(g_bn0(linear(z, gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(g_bn1(
                        linear(h0, gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [batch_size, s_h4, s_w4, gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(g_bn2(
                        deconv2d(h1, [batch_size, s_h2, s_w2, gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [batch_size, s_h, s_w, c_dim], name='g_h3'))


    agent = DCGAN(generator, discriminator, sampler, input_height, input_width, output_height, output_width, batch_size, y_dim, z_dim, gfc_dim, dfc_dim, c_dim, dataset_name, input_fname_pattern, checkpoint_dir, sample_dir)

if __name__ == "__main__":
    test_dcgan()
