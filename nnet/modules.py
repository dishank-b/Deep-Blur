import numpy as np 
import tensorflow as tf


def weight_init(shape, name=None, initializer=tf.contrib.layers.xavier_initializer()):
	"""
	Weights Initialization
	"""
	if name is None:
		name='W'
	
	W = tf.get_variable(name=name, shape=shape, 
		initializer=initializer)

	tf.summary.histogram(name, W)
	return W 


def bias_init(shape, name=None, constant=0.0):
	"""
	Bias Initialization
	"""
	if name is None:
		name='b'

	b = tf.get_variable(name=name, shape=shape, 
		initializer=tf.constant_initializer(constant))

	tf.summary.histogram(name, b)
	return b


def conv2d(input_layer, kernel, out_channels, stride=1, name=None, reuse=False, 
	initializer=tf.contrib.layers.xavier_initializer(), bias_constant=0.01):	
	"""
	2D convolution layer with relu activation
	"""
	if name is None:
		name='2d_convolution'

	in_channels = input_layer.get_shape().as_list()[-1]
	kernel = [kernel, kernel, in_channels, out_channels]
	with tf.variable_scope(name, reuse=reuse):
		W = weight_init(kernel, 'W', initializer)
		b = bias_init(kernel[3], 'b', bias_constant)

		strides=[1, stride, stride, 1]
		output = tf.nn.conv2d(input=input, filter=W, strides=strides, padding='SAME')
		output = output + b
	 	return output


def deconv(input_layer, kernel, out_channels, stride=1, name=None,
	reuse=False, initializer=tf.contrib.layers.xavier_initializer(),
	bias_constant=0.0):
	"""
	2D deconvolution layer with relu activation
	"""
	if name is None:
		name='de_convolution'

	in_channels = input_layer.get_shape().as_list()[-1]
	kernel = [kernel, kernel, out_channels, in_channels]
	with tf.variable_scope(name, reuse=reuse):
		W = weight_init(kernel, 'W', initializer)
		b = bias_init(kernel[2], 'b', bias_constant)

		strides=[1, stride, stride, 1]
		output = tf.nn.conv2d_transpose(value=input, filter=W, output_shape=output_shape, strides=strides)
		output = output + b
		return output


def max_pool(input, kernel=3, stride=2, name=None):
	"""	
	Max-pool
	"""
	if name is None: 
		name='max_pool'

	with tf.variable_scope(name):
		ksize = [1, kernel, kernel, 1]
		strides = [1, stride, stride, 1]
		output = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding='SAME')

		return output


def fully_connected(input, output_neurons, name=None, reuse=False,
	bias_constant=0.01, initializer=tf.contrib.layers.xavier_initializer()):
	"""
	Fully-connected linear activations
	"""
	if name is None:
		name='fully_connected'

	shape = input.get_shape()
	input_units = int(shape[1])
	with tf.variable_scope(name, reuse=reuse):
		W = weight_init([input_units, output_neurons], 'W', initializer)
		b = bias_init([output_neurons], 'b', bias_constant)

		output = tf.add(tf.matmul(input, W), b)
		return output


def dropout_layer(input, keep_prob=0.5, name=None):
	"""
	Dropout layer
	"""
	if name is None:
		name='Dropout'

	with tf.variable_scope(name):
		output = tf.nn.dropout(input, keep_prob=keep_prob)
		return output


def relu(input_layer, name=None):
	"""
	ReLU activation
	"""
	if name is None:
		name = "relu"

	with tf.variable_scope(name):
		tf.nn.relu(input_layer)


def leaky_relu(input, alpha=0.2, name=None):
	"""
	Leaky ReLU
	"""
	if name is None:
		name = "leaky_relu"

	with tf.variable_scope(name):
		o1 = 0.5 * (1 + alpha)
		o2 = 0.5 * (1 - alpha)
		return o1 * input + o2 * abs(input)


def batch_normalize(input_layer, dimension, name=None, use_global=False):
	"""
	Applies batch normalization
	"""
	if name is None:
		name = "BatchNorm"

	if use_global:
		axis = [0, 1, 2]
	else:
		axis = [0]

	with tf.variable_scope(name):
		mean, variance = tf.nn.moments(input_layer, axis=axis)
		beta = tf.get_variable(name="beta", shape=dimension, initializer=tf.constant_initializer(0.0, tf.float32))
		gamma = tf.get_variable(name="gamma", shape=dimension, initializer=tf.constant_initializer(1,0, tf.float32))
		output = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
		return output



def residual_block(input_layer, output_channels, stride=1, first_block=False, name=None, reuse=False):
	"""
	Builds a residual block
	Series of operations include : 
		-> bactch_norm
		-> relu
		-> conv
	"""
	if name is None:
		name = "residual_block"

	input_channels = input_layer.get_shape().as_list()[-2]
	with tf.variable_scope(name):
		# First conv block 
		with tf.variable_scope("block_1"):
			conv1_bn = batch_normalization(input_layer, input_layer.get_shape().as_list()[-1])
			conv1_ac = relu(conv1_bn)
			conv1 = conv2d(conv1_ac, kernel=3, stride=stride, name="conv1", reuse=reuse)
		# Second conv block
		with tf.variable_scope("block_2"):
			conv2_bn = batch_normalization(conv1, conv1.get_shape().as_list()[-1])
			conv2_ac = relu(conv2_bn)
			conv2 = conv2d(conv2_ac, kernel=3, stride=stride, name="conv2", reuse=reuse)
		if conv2.get_shape().as_list()[-1] != input_layer.get_shape().as_list()[-1]:
			raise ValueError('Output and input channels do not match')	  
		else:
			output = input_layer + conv2
		return output
	
