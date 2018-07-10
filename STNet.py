import tensorflow as tf
import tensorflow.contrib.slim as slim
from spatial_transformer import *
import numpy as np
from non_rigid_transformer import *


def STNet(Img, feature, image_size, num_class = 6, stddev = 0.01, dropout_keep_prob = 0.7, reuse = None, scope = 'Spatial_Transform_Nets', net_init = tf.constant([0.6, 0, 0, 0, 0.6, 0])):
    outsize = (int(image_size), int(image_size))
    stl = AffineTransformer(outsize)
    with tf.variable_scope(scope, reuse=reuse):

        net = slim.conv2d(feature, 16, [7, 7], stride = 2, scope = 'conv1',  weights_initializer = tf.random_normal_initializer(stddev=stddev))

        net = slim.conv2d(net, 32, [7, 7], stride = 2, scope = 'conv2',  weights_initializer = tf.random_normal_initializer(stddev=stddev))

        net = slim.max_pool2d(net, [2, 2], [2, 2], scope = 'pool1', padding = 'VALID')

        phi_I = tf.einsum('ijkm,ijkn->imn',net,net)        
        phi_I = tf.reshape(phi_I,[-1,32*32])
        phi_I = tf.divide(phi_I,784.0)  
        y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))
        net = tf.nn.l2_normalize(y_ssqrt, dim=1)
       
        net = slim.fully_connected(net, 64, scope = 'full1', weights_initializer = tf.random_normal_initializer(stddev=stddev))

        #net = slim.dropout(net, dropout_keep_prob)

        net = slim.fully_connected(net, num_class, scope = 'full2', activation_fn = None, weights_initializer = tf.random_normal_initializer(stddev=stddev))

        net = tf.nn.bias_add(net, net_init)
        outputs = tf.reshape(stl.transform(Img, net), [-1, image_size, image_size, 3])
        #outputs = transformer(Aug_inputs, net, [image_size, image_size], name = 'Spatial_Trans')

        return outputs

def Aug(img):
    
    img_rl = tf.image.flip_left_right(img)
    img_ud = tf.image.flip_up_down(img)
    img_tr = tf.image.flip_left_right(img_ud)
    
    out_1 = tf.concat([img_tr, img_ud, img_tr], 1)
    out_2 = tf.concat([img_rl, img, img_rl], 1)
    out_3 = tf.concat([img_tr, img_ud, img_tr], 1)
    '''
    out_1 = tf.concat([img, img, img], 1)
    out_2 = tf.concat([img, img, img], 1)
    out_3 = tf.concat([img, img, img], 1)
    '''
    
    out = tf.concat([out_1, out_2, out_3],0)
    return out


def STNet_arg_scope(weight_decay=0.00005,
                                  batch_norm_decay=0.9997,
                                  batch_norm_epsilon=0.001,
                                  use_batch_norm = False
                                  ):
    '''
    Returns:
        a arg_scopefully_connec with the parameters needed for STNet.
    '''
  # Set weight_decay for weights in conv2d and fully_connected layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay), biases_regularizer=slim.l2_regularizer(weight_decay), activation_fn=tf.nn.relu, biases_initializer=tf.zeros_initializer()):
    
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon
            }
        if use_batch_norm:
            normalizer_fn = slim.batch_norm
            normalizer_params = batch_norm_params
        else:
            normalizer_fn = None
            normalizer_params = {}

        # Set activation_fn and parameters for batch_norm.
        with slim.arg_scope([slim.conv2d], normalizer_fn = normalizer_fn, normalizer_params = normalizer_params, padding = 'VALID') as scope: 
            return scope


        


