# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from Ops import *

class UNET(object):
	"""AFGAN Model"""
	def __init__(self, arg):
		self.arg = arg
	
	def encoder(self, img, code_length):
		with tf.variable_scope("Encoder") as scope:
			E_conv1 = Conv_2D(img, output_chan=32, use_bn=True,name="E_Conv1")
			E_conv2 = Conv_2D(E_conv1, output_chan=64, use_bn=True ,name="E_Conv2")
			E_conv3 = Conv_2D(E_conv2, output_chan=128, use_bn=True ,name="E_Conv3")
			E_conv4 = Conv_2D(E_conv3, output_chan=512, use_bn=True ,name="E_Conv4")
			E_conv4_r = tf.reshape(E_conv4, shape=[-1, int(np.prod(E_conv4.get_shape()[1:]))])

			E_mean = Dense(E_conv4_r, output_dim=code_length, activation=None,use_bn=False, name="E_mean")
			E_signma = Dense(E_conv4_r, output_dim=code_length, activation=None, use_bn=False, name ="E_sigma")
			
			return E_mean, E_sigma

	def generator(self,z):
		with tf.variable_scope("Generator") as scope:
			G_linear1 = Dense(x, output_dim=1024, name="G_hidden1")
			G_linear2 = Dense(G_linear1, output_dim=4*4*512, name="G_hidden2")
			G_linear2_r = tf.reshape(G_linear2, shape=[-1,4, 4, 256])
			G_Dconv3 = Dconv_2D(G_linear2_r, output_chan=128, name="G_hidden3")
			G_Dconv4 = Dconv_2D(G_Dconv3, output_chan=128, name="G_hidden4")
			G_Dconv5 = Dconv_2D(G_Dconv4, output_chan=64, name="G_hidden4")
			G_Dconv6 = Dconv_2D(G_Dconv5, output_chan=32, name="G_hidden4")
			G_Dconv7 = Dconv_2D(G_Dconv6, output_chan=16, name="G_hidden4")
			G_Dconv8 = Dconv_2D(G_Dconv7, output_chan=8, name="G_hidden4")
			G_Dconv9 = Dconv_2D(G_Dconv8, output_chan=3, name="G_output")
			return tf.nn.sigmoid(G_Dconv9)

	def discriminator(self, img):
		with tf.variable_scope("Discriminator") as scope:
			D_conv1  = Conv_2D(img, output_chan=32, use_bn=True, name="D_conv1")
			D_conv2  = Conv_2D(D_conv1, output_chan=64, use_bn=True, name="D_conv1")
			D_conv3  = Conv_2D(D_conv2, output_chan=128, use_bn=True, name="D_conv1")
			D_conv4  = Conv_2D(D_conv3, output_chan=512, use_bn=True, name="D_conv1")
			D_conv4_r = tf.reshape(D_conv4, shape=[-1, int(np.prod(D_conv4.get_shape()[1:]))])
			D_linear5 = Dense(D_conv4_r,output_dim=1028)

			return tf.nn.sigmoid(D_linear5)

	def build_model(self):
		with tf.name_scope("Inputs") as scope:
			self.x_norm = tf.placeholder(tf.float32, shape=[None,512,512,3], name="Input_Normal")
			self.x_blur = tf.placeholder(tf.float32, shape=[None,512,512,3], name="Input_Blurred")
			self.z = tf.placeholder(tf.float32, shape=[None, 100], name="Noise")
			self.train_phase = tf.placeholder(tf.bool, name="is_train")
			self.x_summ = tf.summary.image("Input Images", self.x)
			self.z_summ = tf.summary.histogram("Input Noise", self.z)

		with tf.name_scope("Model") as scope:
			self.encod_mean, self.encod_sigma = self.encoder(self.x_norm, 100)
			self.gen_in = self.encod_mean + self.encod_sigma*self.z, 0, 1, dtype=tf.float32)
			self.gen_out = self.generator(self.gen_in)
			self.dis_real = self.discriminator(self.x_blur, reuse=False)
			self.dis_fake = self.discriminator(self.gen_out, reuse=True)
			self.gen_summ = tf.summary.image("Generator images", self.gen_out)

		with tf.name_scope("Loss") as scope:
			self.marg_likeli = tf.reduce_sum(self.x_blur * tf.log(self.gen_out) + (1 - self.x_blur) * tf.log(1 - self.gen_out), 1)
			self.KL_diver = 0.5 * tf.reduce_sum(tf.square(self.encod_mean) + tf.square(self.encod_sigma) - tf.log(1e-8 + tf.square(self.encod_sigma)) - 1, 1)
			self.marg_likeli = tf.reduce_mean(self.marg_likeli)
			self.KL_diver = tf.reduce_mean(self.KL_diver)
			self.encod_loss = self.KL_diver - self.marg_likeli
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

	def train_model(self,inputs,learning_rate=1e-5, batch_size=64, epoch_size=300):

		with tf.name_scope("Optimizers") as scope:
			D_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.1).minimize(self.dis_loss, var_list=self.d_vars)
			G_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.3).minimize(self.gen_loss, var_list=self.g_vars)
			E_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.1).minimize(self.encod_loss, var_list=self.enc_vars)
		
		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self.writer = tf.summary.FileWriter(self.graph_path)
		self.writer.add_graph(self.sess.graph)

		with tf.name_scope("Training") as scope:

			for epoch in range(epoch_size):
				for itr in xrange(0, len(inputs)-batch_size, batch_size):
					norm_images = inputs[0][itr:itr+batch_size]
					blur_images = inputs[1][itr:itr+batch_size]
					batch_z = np.random_normal(shape=[batch_size,code_length])

					D_inputs = [D_solver ,self.dis_real_loss, self.dis_fake_loss, self.dis_loss]
					D_outputs = self.sess.run(D_inputs, {self.x_norm:norm_images,self.x_blur:blur_images,
						self.z:batch_z, self.train_phase:True})

					G_inputs = [G_solver, E_solver, self.gen_loss, self.encod_loss]
					G_outputs = self.sess.run(G_inputs, {self.x_norm:norm_images,self.x_blur:blur_images,
						self.z:batch_z, self.train_phase:True})

					if itr%5==0:
						print "Epoch: ", epoch, "Iteration: ", itr
						print "Dis Fake Loss: ", D_outputs[3], "Dis Real Loss: ", D_outputs[2], "Dis Total Loss", D_outputs[4]
						print "Generator Loss: ", G_outputs[2]

				if epoch%5==0:
					self.saver.save(self.sess, self.save_path+"/chkpnt")
					print "Checkpoint saved"








