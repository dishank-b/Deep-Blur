# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from ops import *

class UNET(object):
	"""AFGAN Model"""
	def __init__(self, model_path):
		self.graph_path = model_path+"/tf_graph/"
		self.save_path = model_path + "/saved_model/"
		self.output_path = model_path + "/results/"

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
			self.z = tf.placeholder(tf.float32, shape=[None, 100], name="Noise")
			self.train_phase = tf.placeholder(tf.bool, name="is_train")
			self.x_norm_summ = tf.summary.image("Input Images", self.x_norm)
			self.x_blur_summ = tf.summary.image("Input Images", self.x_blur)
			self.z_summ = tf.summary.histogram("Input Noise", self.z)

		with tf.name_scope("Model") as scope:
			self.encod_mean, self.encod_sigma = self.encoder(self.x_norm, 100)
			self.gen_in = self.encod_mean + self.encod_sigma*self.z
			self.gen_out = self.generator(self.gen_in, batch_size)
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
			self.g_vars = [var for var in train_vars if "G_" in var.name]
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

	def train_model(self,inputs,learning_rate=1e-5, batch_size=4, epoch_size=1000000000):

		with tf.name_scope("Training") as scope:
			sample_z = np.random.normal(size=(batch_size,100))
			for epoch in range(epoch_size):
				for itr in xrange(0, len(inputs[0])-batch_size, batch_size):
					norm_images = inputs[0][itr:itr+batch_size]
					blur_images = inputs[1][itr:itr+batch_size]
					batch_z = np.random.normal(size=(batch_size,100))

					D_inputs = [self.D_solver ,self.dis_real_loss, self.dis_fake_loss, self.dis_loss]
					D_outputs = self.sess.run(D_inputs, {self.x_norm:norm_images,self.x_blur:blur_images,
						self.z:batch_z, self.train_phase:True})

					G_inputs = [self.G_solver, self.E_solver, self.gen_loss, self.encod_loss, self.encod_mean, self.encod_sigma, self.E_conv1, self.E_conv4]
					G_outputs = self.sess.run(G_inputs, {self.x_norm:norm_images,self.x_blur:blur_images,
						self.z:batch_z, self.train_phase:True})

					if itr%5==0:
						print "Epoch: ", epoch, "Iteration: ", itr
						print "Dis Fake Loss: ", D_outputs[2], "Dis Real Loss: ", D_outputs[1], "Dis Total Loss", D_outputs[3]
						print "Generator Loss: ", G_outputs[2]
						print "Encoder Loss: ", G_outputs[3]
						# print "Enc conv1: ", G_outputs[-2]
						# print "Enc conv4: ", G_outputs[-1]
						# print 'Mean: {}\tSigma:{}'.format(G_outputs[-4], G_outputs[-3])

				if epoch%10==0:
					self.saver.save(self.sess, self.save_path)
					print "Checkpoint saved"

					input_img = inputs[0][5:9]

					generated_images = self.sess.run([self.gen_out], {self.x_norm: input_img, self.z : sample_z, self.train_phase:False})
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
