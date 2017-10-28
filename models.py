# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import cv2
from Ops import *

class GAN(object):
	"""Model for GAN"""
	def __init__(self, model_path):
		self.model_path = model_path
		self.graph_path = model_path+"/tf_graph"
		self.results_path = model_path+"/results"
		self.save_path = model_path+"/saved_model"
		self.dis_loss = 0.0
		self.gen_loss = 0.0
		self.dis_fake_loss = 0.0
		self.dis_real_loss = 0.0
		self.d_vars = []
		self.g_vars = []

	def discriminator(self,x, reuse=False): # input will always we a image
		with tf.variable_scope("Discriminator", reuse=reuse) as scope:
			D_conv1 = Conv_2D(x, output_chan=16, activation=None, use_bn=False,name="D_Conv1")
			D_conv2 = Conv_2D(D_conv1, output_chan=32,activation=None,use_bn=False ,name="D_Conv2")
			D_conv3 = Conv_2D(D_conv2, output_chan=64, activation=None, use_bn=False,name="D_Conv3")
			D_conv4 = Conv_2D(D_conv3, output_chan=128, activation=None, use_bn=False,name="D_Conv4")
			D_conv4_r = tf.reshape( D_conv4, shape=[-1, int(np.prod(D_conv4.get_shape()[1:]))])
			D_linear5 = Dense(D_conv4_r, output_dim=128, activation=None, use_bn=False, name="D_linear5")
			return tf.sigmoid(D_linear5)

	def generator(self, x):
		with tf.variable_scope("Generator") as scope:
			G_linear1 = Dense(x, output_dim=1024, name="G_hidden1")
			G_linear2 = Dense(G_linear1, output_dim=4*4*128, name="G_hidden2")
			G_linear2_r = tf.reshape(G_linear2, shape=[-1,4, 4, 128])
			G_Dconv3 = Dconv_2D(G_linear2_r, output_chan=64, name="G_hidden3")
			G_Dconv4 = Dconv_2D(G_Dconv3, output_chan=32, name="G_hidden4")
			G_Dconv5 = Dconv_2D(G_Dconv4, output_chan=3, name="G_output")
			return tf.nn.sigmoid(G_Dconv5)

	def Build_Model(self):
		with tf.name_scope("Inputs") as scope:
			self.x = tf.placeholder(tf.float32, shape=[None,32,32,3], name="Input")
			self.z = tf.placeholder(tf.float32, shape=[None, 100], name="Noise")
			self.train_phase = tf.placeholder(tf.bool, name="is_train")
			self.x_summ = tf.summary.image("Input Images", self.x)
			self.z_summ = tf.summary.histogram("Input Noise", self.z)

		with tf.name_scope("Model") as scope:
			self.gen_out = self.generator(self.z)
			self.dis_real = self.discriminator(self.x, reuse=False)
			self.dis_fake = self.discriminator(self.gen_out, reuse=True)
			self.gen_summ = tf.summary.image("Generator images", self.gen_out)

		with tf.name_scope("Loss") as scope:
			self.dis_fake_loss = tf.reduce_mean(-tf.log(self.dis_real))
			self.dis_real_loss = tf.reduce_mean(-tf.log(1- self.dis_fake))
			self.dis_loss = self.dis_real_loss + self.dis_fake_loss
			self.gen_loss = tf.reduce_mean(-tf.log(self.dis_fake))
			self.dis_loss_summ = tf.summary.scalar("Discriminator Loss", self.dis_loss)
			self.gen_loss_summ = tf.summary.scalar("Generator Loss", self.gen_loss)

			train_vars = tf.trainable_variables()
			self.d_vars = [var for var in train_vars if "D_" in var.name]
			self.g_vars = [var for var in train_vars if "G_" in var.name]

	def Train_Model(self,inputs, learning_rate=1e-5, batch_size=64, epoch_size=100):
		with tf.name_scope("Optimizers") as scope:
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # Important thing when using batch_norm to update population mean and variance
    		with tf.control_dependencies(update_ops):               # If doesn't want to use this then, use None in update_collection param of batch_norm	s
				D_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.1).minimize(self.dis_loss, var_list=self.d_vars)
				G_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.3).minimize(self.gen_loss, var_list=self.g_vars)

		self.G_summ = tf.summary.merge([self.z_summ, self.gen_summ, self.gen_loss_summ])
		self.D_summ = tf.summary.merge([self.z_summ, self.x_summ, self.dis_loss_summ])

		self.sess = tf.Session()
		self.saver = tf.train.Saver()
		self.sess.run(tf.global_variables_initializer())
		self.writer = tf.summary.FileWriter(self.graph_path)
		self.writer.add_graph(self.sess.graph)

		with tf.name_scope("Training") as scope:

			for epoch in range(epoch_size):
				for itr in xrange(0, len(inputs)-batch_size, batch_size):
					batch_images = inputs[itr:itr+batch_size]
					batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))

					D_inputs = [D_solver, self.D_summ ,self.dis_real_loss, self.dis_fake_loss, self.dis_loss]
					D_outputs = self.sess.run(D_inputs, {self.x:batch_images, self.z:batch_z, self.train_phase:True})
					self.writer.add_summary(D_outputs[1])

					G_inputs = [G_solver, self.G_summ, self.gen_loss]
					G_outputs = self.sess.run(G_inputs, {self.z:batch_z, self.train_phase:True})
					self.writer.add_summary(G_outputs[1])

					if itr%5==0:
						print "Epoch: ", epoch, "Iteration: ", itr
						print "Dis Fake Loss: ", D_outputs[3], "Dis Real Loss: ", D_outputs[2], "Dis Total Loss", D_outputs[4]
						print "Generator Loss: ", G_outputs[2]

				if epoch%5==0:
					self.saver.save(self.sess, self.save_path+"/checkpoint")
					print "Checkpoint saved"

					sample_z = np.random.uniform(-1,1,size=(batch_size, 100))
					generated_images = self.sess.run([self.gen_out], {self.z : sample_z, self.train_phase:False})
					all_images = np.array(generated_images[0])
					
					for i in range(8):
						image_grid_horizontal = 255.0*all_images[i*24]
						for j in range(7):
							image = 255.0*all_images[i*24+(j+1)*3]
							image_grid_horizontal = np.hstack((image_grid_horizontal, image))
						if i==0:
							image_grid_vertical = image_grid_horizontal
						else:
							image_grid_vertical = np.vstack((image_grid_vertical, image_grid_horizontal))

					cv2.imwrite(self.results_path +"/img_"+str(epoch)+".jpg", image_grid_vertical)








