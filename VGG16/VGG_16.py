import numpy as np
import tensorflow as tf
import time 
import os

VGG_MEAN = [103.939, 116.779, 123.68]

class VGG_16(object):
    def __init__(self,weights_path=None):
        if weights_path==None:
            path = './vgg16.npy'
            vgg_16_weights_path = path
            print(vgg_16_weights_path)
        self.data_dict = np.load(vgg_16_weights_path, encoding='latin1').item()

    def build(self,rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        print(rgb_scaled)
        # Convert RGB to BGR
        red, green, blue = tf.split(rgb_scaled,3,3)
        print (red.get_shape().as_list())
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        print(blue - VGG_MEAN[0],'blue_shape')
        bgr = tf.concat( [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]], 3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 =self.conv_layer(bgr, name='conv1_1')
        self.conv1_2 =self.conv_layer(self.conv1_1,name='conv1_2')
        self.pool1 = self.max_pool_layer(self.conv1_2, name='pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool_layer(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool_layer(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool_layer(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool_layer(self.conv5_3, 'pool5')
        
        self.fc6 = self.fc_layer(self.pool5,name='fc6')
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)  
        self.fc7 = self.fc_layer(self.relu6,name='fc7')
        self.relu7 = tf.nn.relu(self.fc7)
        self.fc8 = self.fc_layer(self.relu7,name = 'fc8')
        assert self.fc8.get_shape().as_list()[1:] == [1000]

        self.prob = tf.nn.softmax(self.fc8,name ='prob')
        self.data_dict = None
        print('build time is {}'.format(time.time()-start_time))

    def conv_layer(self,bottom,name):
        with tf.variable_scope(name):
            filt =self.get_conv_filter(name)
            conv_value  =tf.nn.conv2d(bottom,filt,[1,1,1,1],padding='SAME')
            conv_bias = self.get_bias(name)
            conv_out_value = tf.nn.bias_add(conv_value,conv_bias)
            relu_value = tf.nn.relu(conv_out_value)
            return relu_value

    def max_pool_layer(self,bottom,name):
        return tf.nn.max_pool(bottom,[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=name)

    def fc_layer(self,bottom,name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list() # a list
            dim = 1
            for d in shape[1:]: 
                dim *=d
            x = tf.reshape(bottom,[-1,dim])
            weights = self.get_fc_weights(name)
            bias = self.get_bias(name)
            fc_out = tf.nn.bias_add(tf.matmul(x,weights),bias)

            return fc_out

    def get_conv_filter(self,name):
        return tf.constant(self.data_dict[name][0],name='filter_weights')

    def get_fc_weights(self,name):
        return tf.constant(self.data_dict[name][0],name='fc_weights')

    def get_bias(self,name):
        return tf.constant(self.data_dict[name][1],name='biases')



