from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=3, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None,
                 root_dir=None, gen_graph=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.gen_graph = gen_graph

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # Dis1 => Discriminates depth map
        self.d1_bn1 = batch_norm(name='d1_bn1')
        self.d1_bn2 = batch_norm(name='d1_bn2')
        self.d1_bn3 = batch_norm(name='d1_bn3')

        # Dis2 => Discriminates blurred image
        self.d2_bn1 = batch_norm(name='d2_bn1')
        self.d2_bn2 = batch_norm(name='d2_bn2')
        self.d2_bn3 = batch_norm(name='d2_bn3')

        # Gen1 => Generates the depth map
        self.g1_bn_e2 = batch_norm(name='g1_bn_e2') # Encoder
        self.g1_bn_e3 = batch_norm(name='g1_bn_e3')
        self.g1_bn_e4 = batch_norm(name='g1_bn_e4')
        self.g1_bn_e5 = batch_norm(name='g1_bn_e5')
        self.g1_bn_e6 = batch_norm(name='g1_bn_e6')
        self.g1_bn_e7 = batch_norm(name='g1_bn_e7')
        self.g1_bn_e8 = batch_norm(name='g1_bn_e8')
        self.g1_bn_d1 = batch_norm(name='g1_bn_d1') # Decoder
        self.g1_bn_d2 = batch_norm(name='g1_bn_d2')
        self.g1_bn_d3 = batch_norm(name='g1_bn_d3')
        self.g1_bn_d4 = batch_norm(name='g1_bn_d4')
        self.g1_bn_d5 = batch_norm(name='g1_bn_d5')
        self.g1_bn_d6 = batch_norm(name='g1_bn_d6')
        self.g1_bn_d7 = batch_norm(name='g1_bn_d7')

        # Gen2 => Generates the blurred image
        self.g2_bn_e2 = batch_norm(name='g2_bn_e2') # Encoder
        self.g2_bn_e3 = batch_norm(name='g2_bn_e3')
        self.g2_bn_e4 = batch_norm(name='g2_bn_e4')
        self.g2_bn_e5 = batch_norm(name='g2_bn_e5')
        self.g2_bn_e6 = batch_norm(name='g2_bn_e6')
        self.g2_bn_e7 = batch_norm(name='g2_bn_e7')
        self.g2_bn_e8 = batch_norm(name='g2_bn_e8')
        self.g2_bn_d1 = batch_norm(name='g2_bn_d1') # Decoder
        self.g2_bn_d2 = batch_norm(name='g2_bn_d2')
        self.g2_bn_d3 = batch_norm(name='g2_bn_d3')
        self.g2_bn_d4 = batch_norm(name='g2_bn_d4')
        self.g2_bn_d5 = batch_norm(name='g2_bn_d5')
        self.g2_bn_d6 = batch_norm(name='g2_bn_d6')
        self.g2_bn_d7 = batch_norm(name='g2_bn_d7')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        with open(os.path.join(root_dir, 'data', 'train.txt')) as f:
            self.train_list = f.readlines()
        with open(os.path.join(root_dir, 'data', 'val.txt')) as f:
            self.val_list = f.readlines()
        self.train_list = [path.strip() for path in self.train_list]
        self.val_list = [path.strip() for path in self.val_list]


    def build_model(self):
        with tf.variable_scope("input"):
            self.real_normal = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size, 3],
                                        name='real_normal')
            self.real_depth = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size, 1],
                                            name='real_depth')
            self.real_blur = tf.placeholder(tf.float32,
                                            [self.batch_size, self.image_size, self.image_size, 3],
                                            name='real_blur')

        # Generate Depth map
        self.fake_depth = self.generator_depth(self.real_normal)

        # Generate blur images conditioned on the normal image and depth map
        gen2_input = tf.concat([self.real_normal, self.fake_depth], axis=3)
        self.fake_blur = self.generator_blur(gen2_input)

        # TODO: Currently using unconditional discriminator
        # Depth discriminator
        self.Dd, self.Dd_logits = self.discriminator_depth(self.real_depth, reuse=False)  # Real
        self.Dd_, self.Dd_logits_ = self.discriminator_depth(self.fake_depth, reuse=True) # Fake
        # Blur discriminator
        self.Db, self.Db_logits = self.discriminator_blur(self.real_blur, reuse=False)  # Real
        self.Db_, self.Db_logits_ = self.discriminator_blur(self.fake_blur, reuse=True) # Fake

        # if self.gen_graph != 'graph':
        #     # Sample Depth map images and blur images
        #     self.fake_depth_sample = self.sampler_depth(self.real_normal)
        #     gen2_input = tf.concat([self.real_normal, self.fake_depth_sample], axis=3)
        #     self.fake_blur_sample = self.sampler_blur(gen2_input)

        # Summaries
        self.d1_sum = tf.summary.histogram("d1_real", self.Dd)
        self.d1__sum = tf.summary.histogram("d1_fake_", self.Dd_)
        self.fake_depth_sum = tf.summary.image("fake_depth", self.fake_depth)

        self.d2_sum = tf.summary.histogram("d2_real", self.Db)
        self.d2__sum = tf.summary.histogram("d2_fake_", self.Db_)
        self.fake_blur_sum = tf.summary.image("fake_blur", self.fake_blur)

        # Depth map related loss (TODO: Complete g1_loss, should use L2 or L1 ?)
        with tf.variable_scope("loss"):
            with tf.variable_scope("depth_loss"):
                # Blur image related loss
                with tf.variable_scope("loss_real"):
                    self.d1_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dd_logits, labels=tf.ones_like(self.Dd)))
                with tf.variable_scope("loss_fake"):
                    self.d1_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dd_logits_, labels=tf.zeros_like(self.Dd_)))
                with tf.variable_scope("gen_loss"):
                    self.g1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dd_logits_, labels=tf.ones_like(self.Dd_)))
                with tf.variable_scope("dis_loss"):
                    self.d1_loss = self.d1_loss_real + self.d1_loss_fake

            with tf.variable_scope("blur_loss"):
                # Blur image related loss
                with tf.variable_scope("loss_real"):
                    self.d2_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Db_logits, labels=tf.ones_like(self.Db)))
                with tf.variable_scope("loss_fake"):
                    self.d2_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Db_logits_, labels=tf.zeros_like(self.Db_)))
                with tf.variable_scope("gen_loss"):
                    self.g2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Db_logits_, labels=tf.ones_like(self.Db_))) \
                                    + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_blur - self.fake_blur))
                with tf.variable_scope("dis_loss"):
                    self.d2_loss = self.d2_loss_real + self.d2_loss_fake

        self.d1_loss_real_sum = tf.summary.scalar("d1_loss_real", self.d1_loss_real)
        self.d1_loss_fake_sum = tf.summary.scalar("d1_loss_fake", self.d1_loss_fake)

        self.d2_loss_real_sum = tf.summary.scalar("d2_loss_real", self.d2_loss_real)
        self.d2_loss_fake_sum = tf.summary.scalar("d2_loss_fake", self.d2_loss_fake)

        self.g1_loss_sum = tf.summary.scalar("g1_loss", self.g1_loss)
        self.d1_loss_sum = tf.summary.scalar("d1_loss", self.d1_loss)
        self.g2_loss_sum = tf.summary.scalar("g2_loss", self.g2_loss)
        self.d2_loss_sum = tf.summary.scalar("d2_loss", self.d2_loss)

        t_vars = tf.trainable_variables()

        # Depth map related gen
        self.d1_vars = [var for var in t_vars if 'dis_depth' in var.name]
        self.g1_vars = [var for var in t_vars if 'gen_depth' in var.name]

        # Blur image related gen
        self.d2_vars = [var for var in t_vars if 'dis_blur' in var.name]
        self.g2_vars = [var for var in t_vars if 'gen_blur' or 'gen_depth' in var.name]

        if self.gen_graph != "graph":
            self.saver = tf.train.Saver()


    def load_random_samples(self):
        np.random.shuffle(self.val_list)

        real_A = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        real_B = np.zeros((self.batch_size, self.image_size, self.image_size, 1))
        real_C = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        for idx, path in enumerate(self.val_list[:self.batch_size]):
            path_A, path_B, path_C = path.split(' ')
            real_A[idx] = read_image(path_A)
            real_B[idx] = np.expand_dims(read_image(path_B), -1)
            real_C[idx] = read_image(path_C)
        return real_A, real_B, real_C


    def load_batch(self, begin, end):
        if begin == 0:
            np.random.shuffle(self.train_list)

        real_A = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        real_B = np.zeros((self.batch_size, self.image_size, self.image_size, 1))
        real_C = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        for idx, path in enumerate(self.train_list[begin:end]):
            path_A, path_B, path_C = path.split(' ')
            real_A[idx] = read_image(path_A)
            real_B[idx] = np.expand_dims(read_image(path_B), -1)
            real_C[idx] = read_image(path_C)
        return real_A, real_B, real_C


    def sample_model(self, sample_dir, epoch, idx):
        A, B, C = self.load_random_samples()
        feed_dict = {
            self.real_normal: A,
            self.real_depth: B,
            self.real_blur: C
        }

        # Sample Depth map and blur images
        samples_depth, samples_blur, d1_loss, \
        d2_loss, g1_loss, g2_loss = self.sess.run( \
            [self.fake_depth, self.fake_blur, \
            self.d1_loss, self.d2_loss, self.g1_loss, self.g2_loss], \
            feed_dict=feed_dict
        )
        Bp = np.zeros((len(B), 256, 256))
        Sp = np.zeros((len(samples_depth), 256, 256))
        for idx_val, img in enumerate(B):
            Bp[idx_val] = np.squeeze(B[idx_val])
        for idx_val, img in enumerate(samples_depth):
            Sp[idx_val] = np.squeeze(samples_depth[idx_val])

        save_images(A, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}_real_normal.png'.format(sample_dir, epoch, idx))
        save_images(Bp, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}_real_depth.png'.format(sample_dir, epoch, idx), "gray")
        save_images(C, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}_real_blur.png'.format(sample_dir, epoch, idx))
        save_images(Sp, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}_fake_depth.png'.format(sample_dir, epoch, idx), "gray")
        save_images(samples_blur, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}_fake_blur.png'.format(sample_dir, epoch, idx))
        print("[Sample] d1_loss: {:.8f}, g1_loss: {:.8f} d2_loss: {:.8f}, g2_loss: {:.8f}".format(d1_loss, g1_loss, \
            d2_loss, g2_loss))

    def train(self, args):
        """Train pix2pix"""

        def count_params():
            total_parameters = 0
            for variable in tf.trainable_variables():
                count = 1
                for dimension in variable.get_shape().as_list():
                    count *= dimension
                total_parameters += count

            return total_parameters
        print 'Total parameters: {}'.format(count_params())

        # G1
        d1_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d1_loss, var_list=self.d1_vars)
        g1_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g1_loss, var_list=self.g1_vars)
        # G2
        d2_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d2_loss, var_list=self.d2_vars)
        g2_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g2_loss, var_list=self.g2_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d1__sum, self.d2__sum,
            self.fake_depth_sum, self.fake_blur_sum, \
            self.d1_loss_fake_sum, self.g1_loss_sum, \
            self.d2_loss_fake_sum, self.g2_loss_sum])
        self.d_sum = tf.summary.merge([self.d1_sum, self.d2_sum, \
            self.d1_loss_real_sum, self.d1_loss_sum, \
            self.d2_loss_real_sum, self.d2_loss_sum])
        self.generate_graph()

        counter = 1
        args.train_size = len(self.train_list)
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            batch_num = 0
            for batch_begin, batch_end in zip(xrange(0, args.train_size, args.batch_size), \
                xrange(args.batch_size, args.train_size+1, args.batch_size)):

                real_normal, real_depth, real_blur = self.load_batch(batch_begin, batch_end)
                feed_dict = {
                    self.real_normal: real_normal,
                    self.real_depth: real_depth,
                    self.real_blur: real_blur
                }

                # Update Depth Discriminator network
                _, summary_str = self.sess.run([d1_optim, self.d_sum],
                                               feed_dict=feed_dict
                                               )
                self.writer.add_summary(summary_str, counter)

                # Update Depth Generator network
                _, summary_str = self.sess.run([g1_optim, self.g_sum],
                                               feed_dict=feed_dict
                                               )
                self.writer.add_summary(summary_str, counter)

                # Update Blur Discriminator network
                _ = self.sess.run(d2_optim, feed_dict=feed_dict)

                # Update Blur Generator network
                _ = self.sess.run(g2_optim, feed_dict=feed_dict)


                # # Run the generator twice?
                # _, summary_str = self.sess.run([g1_optim, self.g_sum],
                #                                feed_dict=feed_dict
                #                                )
                # self.writer.add_summary(summary_str, counter)
                # _ = self.sess.run(g2_optim, feed_dict=feed_dict)

                # Evaluate the G1/D1 losses
                errD1_fake = self.d1_loss_fake.eval(feed_dict=feed_dict)
                errD1_real = self.d1_loss_real.eval(feed_dict=feed_dict)
                errG1 = self.g1_loss.eval(feed_dict=feed_dict)

                # Evaluate the G2/D2 losses
                errD2_fake = self.d2_loss_fake.eval(feed_dict=feed_dict)
                errD2_real = self.d2_loss_real.eval(feed_dict=feed_dict)
                errG2 = self.g2_loss.eval(feed_dict=feed_dict)

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d1_loss: %.8f, g1_loss: %.8f d2_loss: %.8f, g2_loss: %.8f" \
                    % (epoch, batch_num, args.train_size // args.batch_size, \
                       time.time() - start_time, \
                       errD1_fake+errD1_real, errG1, \
                       errD2_fake+errD2_real, errG2 \
                       ) \
                    )

                batch_num += 1

                if np.mod(counter, 200) == 1:
                    self.sample_model(args.sample_dir, epoch, counter)

                if np.mod(counter, 2000) == 2:
                    self.save(args.checkpoint_dir, counter)


    def discriminator_depth(self, image, y=None, reuse=False):

        with tf.variable_scope("dis_depth") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d1_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d1_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d1_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def discriminator_blur(self, image, y=None, reuse=False):

        with tf.variable_scope("dis_blur") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h1 = lrelu(self.d2_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h2 = lrelu(self.d2_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h3 = lrelu(self.d2_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator_depth(self, image, y=None):
        with tf.variable_scope("gen_depth") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            with tf.variable_scope("e1"):
                e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
            with tf.variable_scope("e2"):
                e2 = self.g1_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
                # e2 is (64 x 64 x self.gf_dim*2)
            with tf.variable_scope("e3"):
                e3 = self.g1_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
                # e3 is (32 x 32 x self.gf_dim*4)
            with tf.variable_scope("e4"):
                e4 = self.g1_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
                # e4 is (16 x 16 x self.gf_dim*8)
            with tf.variable_scope("e5"):
                e5 = self.g1_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
                # e5 is (8 x 8 x self.gf_dim*8)
            with tf.variable_scope("e6"):
                e6 = self.g1_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
                # e6 is (4 x 4 x self.gf_dim*8)
            with tf.variable_scope("e7"):
                e7 = self.g1_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
                # e7 is (2 x 2 x self.gf_dim*8)
            with tf.variable_scope("e8"):
                e8 = self.g1_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
                # e8 is (1 x 1 x self.gf_dim*8)

            with tf.variable_scope("d1"):
                self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                    [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
                d1 = tf.nn.dropout(self.g1_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e7], 3)
                # d1 is (2 x 2 x self.gf_dim*8*2)

            with tf.variable_scope("d2"):
                self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(e7),
                    [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
                d2 = tf.nn.dropout(self.g1_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e6], 3)
                # d2 is (4 x 4 x self.gf_dim*8*2)

            with tf.variable_scope("d3"):
                self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                    [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
                d3 = tf.nn.dropout(self.g1_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, e5], 3)
                # d3 is (8 x 8 x self.gf_dim*8*2)

            with tf.variable_scope("d4"):
                self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                    [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
                d4 = self.g1_bn_d4(self.d4)
                d4 = tf.concat([d4, e4], 3)
                # d4 is (16 x 16 x self.gf_dim*8*2)

            with tf.variable_scope("d5"):
                self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                    [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
                d5 = self.g1_bn_d5(self.d5)
                d5 = tf.concat([d5, e3], 3)
                # d5 is (32 x 32 x self.gf_dim*4*2)

            with tf.variable_scope("d6"):
                self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                    [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
                d6 = self.g1_bn_d6(self.d6)
                d6 = tf.concat([d6, e2], 3)
                # d6 is (64 x 64 x self.gf_dim*2*2)

            with tf.variable_scope("d7"):
                self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                    [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
                d7 = self.g1_bn_d7(self.d7)
                d7 = tf.concat([d7, e1], 3)
                # d7 is (128 x 128 x self.gf_dim*1*2)

            with tf.variable_scope("d8"):
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                    [self.batch_size, s, s, 1], name='g_d8', with_w=True)
                # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def generator_blur(self, image, y=None):
        with tf.variable_scope("gen_blur") as scope:

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            with tf.variable_scope("e1"):
                e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
            with tf.variable_scope("e2"):
                e2 = self.g1_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
                # e2 is (64 x 64 x self.gf_dim*2)
            with tf.variable_scope("e3"):
                e3 = self.g1_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
                # e3 is (32 x 32 x self.gf_dim*4)
            with tf.variable_scope("e4"):
                e4 = self.g1_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
                # e4 is (16 x 16 x self.gf_dim*8)
            with tf.variable_scope("e5"):
                e5 = self.g1_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
                # e5 is (8 x 8 x self.gf_dim*8)
            with tf.variable_scope("e6"):
                e6 = self.g1_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
                # e6 is (4 x 4 x self.gf_dim*8)
            with tf.variable_scope("e7"):
                e7 = self.g1_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
                # e7 is (2 x 2 x self.gf_dim*8)
            with tf.variable_scope("e8"):
                e8 = self.g1_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
                # e8 is (1 x 1 x self.gf_dim*8)

            with tf.variable_scope("d1"):
                self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                    [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
                d1 = tf.nn.dropout(self.g1_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e7], 3)
                # d1 is (2 x 2 x self.gf_dim*8*2)

            with tf.variable_scope("d2"):
                self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                    [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
                d2 = tf.nn.dropout(self.g1_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e6], 3)
                # d2 is (4 x 4 x self.gf_dim*8*2)

            with tf.variable_scope("d3"):
                self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                    [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
                d3 = tf.nn.dropout(self.g1_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, e5], 3)
                # d3 is (8 x 8 x self.gf_dim*8*2)

            with tf.variable_scope("d4"):
                self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                    [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
                d4 = self.g1_bn_d4(self.d4)
                d4 = tf.concat([d4, e4], 3)
                # d4 is (16 x 16 x self.gf_dim*8*2)

            with tf.variable_scope("d5"):
                self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                    [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
                d5 = self.g1_bn_d5(self.d5)
                d5 = tf.concat([d5, e3], 3)
                # d5 is (32 x 32 x self.gf_dim*4*2)

            with tf.variable_scope("d6"):
                self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                    [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
                d6 = self.g1_bn_d6(self.d6)
                d6 = tf.concat([d6, e2], 3)
                # d6 is (64 x 64 x self.gf_dim*2*2)

            with tf.variable_scope("d7"):
                self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                    [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
                d7 = self.g1_bn_d7(self.d7)
                d7 = tf.concat([d7, e1], 3)
                # d7 is (128 x 128 x self.gf_dim*1*2)

            with tf.variable_scope("d8"):
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                    [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
                # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def sampler_depth(self, image, y=None):

        with tf.variable_scope("gen_depth") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            with tf.variable_scope("e1"):
                e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
            with tf.variable_scope("e2"):
                e2 = self.g1_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
                # e2 is (64 x 64 x self.gf_dim*2)
            with tf.variable_scope("e3"):
                e3 = self.g1_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
                # e3 is (32 x 32 x self.gf_dim*4)
            with tf.variable_scope("e4"):
                e4 = self.g1_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
                # e4 is (16 x 16 x self.gf_dim*8)
            with tf.variable_scope("e5"):
                e5 = self.g1_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
                # e5 is (8 x 8 x self.gf_dim*8)
            with tf.variable_scope("e6"):
                e6 = self.g1_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
                # e6 is (4 x 4 x self.gf_dim*8)
            with tf.variable_scope("e7"):
                e7 = self.g1_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
                # e7 is (2 x 2 x self.gf_dim*8)
            with tf.variable_scope("e8"):
                e8 = self.g1_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
                # e8 is (1 x 1 x self.gf_dim*8)

            with tf.variable_scope("d1"):
                self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                    [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
                d1 = tf.nn.dropout(self.g1_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e7], 3)
                # d1 is (2 x 2 x self.gf_dim*8*2)

            with tf.variable_scope("d2"):
                self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                    [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
                d2 = tf.nn.dropout(self.g1_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e6], 3)
                # d2 is (4 x 4 x self.gf_dim*8*2)

            with tf.variable_scope("d3"):
                self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                    [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
                d3 = tf.nn.dropout(self.g1_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, e5], 3)
                # d3 is (8 x 8 x self.gf_dim*8*2)

            with tf.variable_scope("d4"):
                self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                    [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
                d4 = self.g1_bn_d4(self.d4)
                d4 = tf.concat([d4, e4], 3)
                # d4 is (16 x 16 x self.gf_dim*8*2)

            with tf.variable_scope("d5"):
                self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                    [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
                d5 = self.g1_bn_d5(self.d5)
                d5 = tf.concat([d5, e3], 3)
                # d5 is (32 x 32 x self.gf_dim*4*2)

            with tf.variable_scope("d6"):
                self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                    [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
                d6 = self.g1_bn_d6(self.d6)
                d6 = tf.concat([d6, e2], 3)
                # d6 is (64 x 64 x self.gf_dim*2*2)

            with tf.variable_scope("d7"):
                self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                    [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
                d7 = self.g1_bn_d7(self.d7)
                d7 = tf.concat([d7, e1], 3)
                # d7 is (128 x 128 x self.gf_dim*1*2)

            with tf.variable_scope("d8"):
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                    [self.batch_size, s, s, 1], name='g_d8', with_w=True)
                # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def sampler_blur(self, image, y=None):

        with tf.variable_scope("gen_blur") as scope:
            scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x input_c_dim)
            with tf.variable_scope("e1"):
                e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
                # e1 is (128 x 128 x self.gf_dim)
            with tf.variable_scope("e2"):
                e2 = self.g1_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
                # e2 is (64 x 64 x self.gf_dim*2)
            with tf.variable_scope("e3"):
                e3 = self.g1_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
                # e3 is (32 x 32 x self.gf_dim*4)
            with tf.variable_scope("e4"):
                e4 = self.g1_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
                # e4 is (16 x 16 x self.gf_dim*8)
            with tf.variable_scope("e5"):
                e5 = self.g1_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
                # e5 is (8 x 8 x self.gf_dim*8)
            with tf.variable_scope("e6"):
                e6 = self.g1_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
                # e6 is (4 x 4 x self.gf_dim*8)
            with tf.variable_scope("e7"):
                e7 = self.g1_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
                # e7 is (2 x 2 x self.gf_dim*8)
            with tf.variable_scope("e8"):
                e8 = self.g1_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
                # e8 is (1 x 1 x self.gf_dim*8)

            with tf.variable_scope("d1"):
                self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                    [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
                d1 = tf.nn.dropout(self.g1_bn_d1(self.d1), 0.5)
                d1 = tf.concat([d1, e7], 3)
                # d1 is (2 x 2 x self.gf_dim*8*2)

            with tf.variable_scope("d2"):
                self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                    [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
                d2 = tf.nn.dropout(self.g1_bn_d2(self.d2), 0.5)
                d2 = tf.concat([d2, e6], 3)
                # d2 is (4 x 4 x self.gf_dim*8*2)

            with tf.variable_scope("d3"):
                self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                    [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
                d3 = tf.nn.dropout(self.g1_bn_d3(self.d3), 0.5)
                d3 = tf.concat([d3, e5], 3)
                # d3 is (8 x 8 x self.gf_dim*8*2)

            with tf.variable_scope("d4"):
                self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                    [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
                d4 = self.g1_bn_d4(self.d4)
                d4 = tf.concat([d4, e4], 3)
                # d4 is (16 x 16 x self.gf_dim*8*2)

            with tf.variable_scope("d5"):
                self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                    [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
                d5 = self.g1_bn_d5(self.d5)
                d5 = tf.concat([d5, e3], 3)
                # d5 is (32 x 32 x self.gf_dim*4*2)

            with tf.variable_scope("d6"):
                self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                    [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
                d6 = self.g1_bn_d6(self.d6)
                d6 = tf.concat([d6, e2], 3)
                # d6 is (64 x 64 x self.gf_dim*2*2)

            with tf.variable_scope("d7"):
                self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                    [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
                d7 = self.g1_bn_d7(self.d7)
                d7 = tf.concat([d7, e1], 3)
                # d7 is (128 x 128 x self.gf_dim*1*2)

            with tf.variable_scope("d8"):
                self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                    [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
                # d8 is (256 x 256 x output_c_dim)

            return tf.nn.tanh(self.d8)

    def generate_graph(self):
        print 'Generating the graph...'
        self.writer = tf.summary.FileWriter("logs", self.sess.graph)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = glob('./datasets/{}/val/*.jpg'.format(self.dataset_name))

        # sort testing input
        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.jpg')[0], sample_files)]
        sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample = [load_data(sample_file, is_test=True) for sample_file in sample_files]

        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images):
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            save_images(samples, [self.batch_size, 1],
                        './{}/test_{:04d}.png'.format(args.test_dir, idx))
