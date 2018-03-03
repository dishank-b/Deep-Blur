# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from ops import *

class UNET(object):
	"""AFGAN Model"""
	def __init__(self, model_path, code_len=8, h=512, w=512):
		self.graph_path = model_path+"/tf_graph/"
		self.save_path = model_path + "/saved_model/"
		self.output_path = model_path + "/results/"

		self.non_lin = {'relu' : lambda x: relu(x, name='relu'),
							 'lrelu': lambda x: lrelu(x, name='lrelu'),
							 'tanh' : lambda x: tanh(x, name='tanh')
							}
		self.code_length = code_len
		self.h = h
		self.w = w

	def encoder(self, img, code_length):
		with tf.variable_scope("Encoder") as scope:
			E_conv1 = Conv_2D(img, output_chan=32, use_bn=True,name="E_Conv1")
			self.E_conv1 = max_pool(E_conv1)
			E_conv2 = Conv_2D(self.E_conv1, output_chan=64, use_bn=True ,name="E_Conv2")
			E_conv2 = max_pool(E_conv2)
			E_conv3 = Conv_2D(E_conv2, output_chan=128, use_bn=True ,name="E_Conv3")
			E_conv3 = max_pool(E_conv3)
			E_conv4 = Conv_2D(E_conv3, output_chan=512, use_bn=False ,name="E_Conv4")
			self.E_conv4 = max_pool(E_conv4)
			E_conv4_r = tf.reshape(self.E_conv4, shape=[-1, int(np.prod(E_conv4.get_shape()[1:]))])

			# TODO: assert the shape of sigma and mean
			E_mean = Dense(E_conv4_r, output_dim=code_length, activation=None,use_bn=False, name="E_mean")
			# E_sigma = Dense(E_conv4_r, output_dim=1, activation=None, use_bn=False, name ="E_sigma")

			# Get the variance in the log space
			E_log_var = Dense(E_conv4_r, output_dim=code_length, activation=None, use_bn=False, name ="E_sigma")
			E_sigma = tf.exp(0.5 * E_log_var)

			return E_mean, E_sigma
			# return E_mean, E_log_var

	def generator(self,z,batch_size):
		with tf.variable_scope("Generator") as scope:
			G_linear1 = Dense(z, output_dim=1024, name="G_hidden1")
			G_linear2 = Dense(G_linear1, output_dim=4*4*512, name="G_hidden2")
			G_linear2_r = tf.reshape(G_linear2, shape=[-1,4, 4,512])
			G_Dconv3 = Dconv_2D(G_linear2_r, output_chan=256,batch_size=batch_size ,name="G_hidden3")
			G_Dconv4 = Dconv_2D(G_Dconv3, output_chan=128,batch_size=batch_size , name="G_hidden4")
			G_Dconv5 = Dconv_2D(G_Dconv4, output_chan=64,batch_size=batch_size , name="G_hidden5")
			G_Dconv6 = Dconv_2D(G_Dconv5, output_chan=32,batch_size=batch_size , name="G_hidden6")
			G_Dconv7 = Dconv_2D(G_Dconv6, output_chan=16,batch_size=batch_size , name="G_hidden7")
			G_Dconv8 = Dconv_2D(G_Dconv7, output_chan=8,batch_size=batch_size , name="G_hidden8")
			G_Dconv9 = Dconv_2D(G_Dconv8, output_chan=3,batch_size=batch_size , name="G_output", use_bn=False)
			return tf.nn.tanh(G_Dconv9)


	def unet_gen(self, z, image, batch_size, non_lin='lrelu', reuse=False):

		with tf.name_scope('replication'):
			tiled_z = tf.tile(z, [1, self.w*self.h], name='tiling')
			reshaped = tf.reshape(tiled_z, [-1, self.h, self.w, self.code_length], name='reshape')
			in_layer = tf.concat([image, reshaped], axis=3, name='concat')

		# Downsample
		with tf.variable_scope('down_0'):
			conv0 = conv2d(in_layer, ksize=3, out_channels=16, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv0 = conv2d(conv0,    ksize=3, out_channels=16, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)
			pool0 = max_pool(conv0, kernel=2, stride=2, name='pool1') #256

		with tf.variable_scope('down_1'):
			conv1 = conv2d(pool0, ksize=3, out_channels=32, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv1 = conv2d(conv1,    ksize=3, out_channels=32, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)
			pool1 = max_pool(conv1, kernel=2, stride=2, name='pool1') #128

		with tf.variable_scope('down_2'):
			conv2 = conv2d(pool1, ksize=3, out_channels=64, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv2 = conv2d(conv2, ksize=3, out_channels=64, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)
			pool2 = max_pool(conv2, kernel=2, stride=2, name='pool1') #64

		with tf.variable_scope('down_3'):
			conv3 = conv2d(pool2, ksize=3, out_channels=128, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv3 = conv2d(conv3, ksize=3, out_channels=128, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)
			pool3 = max_pool(conv3, kernel=2, stride=2, name='pool1') #32

		with tf.variable_scope('down_4'):
			conv4 = conv2d(pool3, ksize=3, out_channels=256, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv4 = conv2d(conv4, ksize=3, out_channels=256, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)
			pool4 = max_pool(conv4, kernel=2, stride=2, name='pool1') #16

		with tf.variable_scope('down_5'):
			conv5  = conv2d(pool4, ksize=3, out_channels=512, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv5 = conv2d(conv5,  ksize=3, out_channels=512, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

		# Upsample
		with tf.variable_scope('up_1'):
			dcnv1 = deconv(conv5, ksize=3, out_channels=512, stride=2, name='dconv1', out_shape=32, non_lin=self.non_lin[non_lin],
								batch_size=batch_size, reuse=reuse)
			up1   = concatenate(dcnv1, conv4, axis=3)
			conv6 = conv2d(up1,   ksize=3, out_channels=256, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv6 = conv2d(conv6, ksize=3, out_channels=256, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

		with tf.variable_scope('up_2'):
			dcnv2 = deconv(conv6, ksize=3, out_channels=256, stride=2, name='dconv1', out_shape=64, non_lin=self.non_lin[non_lin],
							  batch_size=batch_size, reuse=reuse)
			up2   = concatenate(dcnv2, conv3, axis=3)
			conv7 = conv2d(up2,   ksize=3, out_channels=128, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv7 = conv2d(conv7, ksize=3, out_channels=128, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

		with tf.variable_scope('up_3'):
			dcnv3 = deconv(conv7, ksize=3, out_channels=128, stride=2, name='dconv1', out_shape=128, non_lin=self.non_lin[non_lin],
							  batch_size=batch_size, reuse=reuse)
			up2   = concatenate(dcnv3, conv2, axis=3)
			conv8 = conv2d(up2,   ksize=3, out_channels=64, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv8 = conv2d(conv8, ksize=3, out_channels=64, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

		with tf.variable_scope('up_4'):
			dcnv4 = deconv(conv8, ksize=3, out_channels=64, stride=2, name='dconv1', out_shape=256, non_lin=self.non_lin[non_lin],
							  batch_size=batch_size, reuse=reuse)
			up3   = concatenate(dcnv4, conv1, axis=3)
			conv9 = conv2d(up3,   ksize=3, out_channels=32, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv9 = conv2d(conv9, ksize=3, out_channels=32, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

		with tf.variable_scope('up_5'):
			dcnv5 = deconv(conv9, ksize=3, out_channels=32, stride=2, name='dconv1', out_shape=512, non_lin=self.non_lin[non_lin],
							  batch_size=batch_size, reuse=reuse)
			up4   = concatenate(dcnv5, conv0, axis=3)
			conv9 = conv2d(up4,   ksize=3, out_channels=16, stride=1, name='conv1', non_lin=self.non_lin[non_lin], reuse=reuse)
			conv9 = conv2d(conv9, ksize=3, out_channels=16, stride=1, name='conv2', non_lin=self.non_lin[non_lin], reuse=reuse)

		with tf.variable_scope('output'):
			output = conv2d(conv9, ksize=3, out_channels=3, stride=1, name='conv1', non_lin=self.non_lin['tanh'], reuse=reuse)

		return output

	def discriminator(self, img, reuse=False):
		with tf.variable_scope("Discriminator", reuse=reuse) as scope:
			D_conv1  = Conv_2D(img, output_chan=32, use_bn=True, name="D_conv1")
			D_conv1 = max_pool(D_conv1)
			D_conv2  = Conv_2D(D_conv1, output_chan=64, use_bn=True, name="D_conv2")
			D_conv2 = max_pool(D_conv2)
			D_conv3  = Conv_2D(D_conv2, output_chan=128, use_bn=True, name="D_conv3")
			D_conv3 = max_pool(D_conv3)
			D_conv4  = Conv_2D(D_conv3, output_chan=512, use_bn=False, name="D_conv4")
			D_conv4 = max_pool(D_conv4)
			# D_conv4_r = tf.reshape(D_conv2, shape=[-1, int(np.prod(D_conv2.get_shape().as_list()[1:]))])
			D_conv4_r = tf.reshape(D_conv4, shape=[-1, int(np.prod(D_conv4.get_shape()[1:]))])
			D_linear5 = Dense(D_conv4_r,output_dim=1028, name="D_dense5")
			D_linear6 = Dense(D_linear5,output_dim=1, name="D_dense6")

			return tf.nn.sigmoid(D_linear6)

	def build_model(self, batch_size=4):
		with tf.name_scope("Inputs") as scope:
			self.x_norm = tf.placeholder(tf.float32, shape=[None,512,512,3], name="Input_Normal")
			self.x_blur = tf.placeholder(tf.float32, shape=[None,512,512,3], name="Input_Blurred")
			self.z = tf.placeholder(tf.float32, shape=[None, self.code_length], name="Noise")
			self.train_phase = tf.placeholder(tf.bool, name="is_train")
			self.x_norm_summ = tf.summary.image("Input Images", self.x_norm)
			self.x_blur_summ = tf.summary.image("Input Images", self.x_blur)
			self.z_summ = tf.summary.histogram("Input Noise", self.z)

		with tf.name_scope("Model") as scope:
			self.encod_mean, self.encod_sigma = self.encoder(self.x_blur, self.code_length)
			self.gen_in = self.encod_mean + self.encod_sigma*self.z
			# self.gen_out = self.generator(self.gen_in, batch_size)

			def train_mode():
				return self.gen_in

			def test_mode():
				return self.z

			self.gen_input_noise = tf.cond(tf.equal(self.train_phase, tf.constant(True)),
													 true_fn=train_mode,
													 false_fn=test_mode,
													 name='Noise'
													 )
			with tf.variable_scope("Generator") as scope:
				self.gen_out = self.unet_gen(self.gen_input_noise, self.x_norm, batch_size)
			self.dis_real = self.discriminator(self.x_blur, reuse=False)
			self.dis_fake = self.discriminator(self.gen_out, reuse=True)
			self.gen_summ = tf.summary.image("Generator images", self.gen_out)

		with tf.name_scope("Loss") as scope:
			# self.marg_likeli = tf.reduce_sum(self.x_blur * tf.log(self.gen_out) + (1 - self.x_blur) * tf.log(1 - self.gen_out), 1)
			self.KL_diver = 0.5 * tf.reduce_sum(tf.square(self.encod_mean) + tf.square(self.encod_sigma) - tf.log(1e-8 + tf.square(self.encod_sigma)) - 1, 1)
			# self.marg_likeli = tf.reduce_mean(self.marg_likeli)
			self.KL_diver = tf.reduce_mean(self.KL_diver, axis=0)
			self.encod_loss = self.KL_diver# - self.marg_likeli

			self.dis_real_loss = tf.reduce_mean(-tf.log(self.dis_real))
			self.dis_fake_loss = tf.reduce_mean(-tf.log(1-self.dis_fake))

			self.dis_loss = self.dis_real_loss + self.dis_fake_loss
			self.gen_loss = tf.reduce_mean(-tf.log(self.dis_fake))

			self.dis_loss_summ = tf.summary.scalar("Discriminator Loss", self.dis_loss)
			self.gen_loss_summ = tf.summary.scalar("Generator Loss", self.gen_loss)

			train_vars = tf.trainable_variables()
			self.d_vars = [var for var in train_vars if "D_" in var.name]
			self.g_vars = [var for var in train_vars if "Generator" in var.name]
			self.enc_vars = [var for var in train_vars if "E_" in var.name]

		with tf.name_scope("Optimizers") as scope:
			self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.1).minimize(self.dis_loss, var_list=self.d_vars)
			self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.3).minimize(self.gen_loss, var_list=self.g_vars)
			self.E_solver = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.1).minimize(self.encod_loss, var_list=self.enc_vars)

		self.sess = tf.Session()
		self.writer = tf.summary.FileWriter(self.graph_path)
		self.writer.add_graph(self.sess.graph)
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())

	def train_model(self,learning_rate, batch_size, epoch_size,inputs):

		with tf.name_scope("Training") as scope:
			sample_z = np.random.normal(size=(batch_size, self.code_length))
			for epoch in range(epoch_size):
				for itr in xrange(0, len(inputs[0])-batch_size, batch_size):
					norm_images = inputs[0][itr:itr+batch_size]
					blur_images = inputs[1][itr:itr+batch_size]
					batch_z = np.random.normal(size=(batch_size, self.code_length))

					D_inputs = [self.D_solver ,self.dis_real_loss, self.dis_fake_loss, self.dis_loss]
					D_outputs = self.sess.run(D_inputs, {self.x_norm:norm_images,self.x_blur:blur_images,
						self.z:batch_z, self.train_phase:True})

					G_inputs = [self.G_solver, self.E_solver, self.gen_loss, self.encod_loss, self.encod_mean, self.encod_sigma, self.E_conv1, self.E_conv4]
					G_outputs = self.sess.run(G_inputs, {self.x_norm:norm_images,self.x_blur:blur_images,
						self.z:batch_z, self.train_phase:True})

					if itr%2==0:
						print "Epoch: {:4d} Iteration: {:3d} D_fake_loss: {:.5f} D_real_loss: {:.5f} D_loss: {:.5f} G_loss: {:.5f} E_loss: {:.5f}".format(
							epoch, itr, D_outputs[2], D_outputs[1], D_outputs[3], G_outputs[2], G_outputs[3])
						# print "Enc conv1: ", G_outputs[-2]
						# print "Enc conv4: ", G_outputs[-1]
						# print 'Mean: {}\tSigma:{}'.format(G_outputs[-4], G_outputs[-3])

				if epoch%10==0:
					self.saver.save(self.sess, self.save_path)
					print "Checkpoint saved"

					input_img = inputs[0][5:9]

					generated_images = self.sess.run([self.gen_out], {self.train_phase:False, self.x_norm: input_img, self.z : sample_z, self.x_blur:blur_images})
					all_images = np.array(generated_images[0])
					
					for i in range(2):
						image_grid_horizontal = 255.0*all_images[i*2]
						for j in range(1):
							image = 255.0*all_images[i*2+j+1]
							image_grid_horizontal = np.hstack((image_grid_horizontal, image))
						if i==0:
							image_grid_vertical = image_grid_horizontal
						else:
							image_grid_vertical = np.vstack((image_grid_vertical, image_grid_horizontal))

					cv2.imwrite(self.output_path +"/img_"+str(epoch)+".jpg", image_grid_vertical)
