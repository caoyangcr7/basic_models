from keras.models import Model  
from keras.layers import Input,Dense,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,ZeroPadding2D ,LeakyReLU
from keras.layers import add,Flatten   
from keras.optimizers import SGD  
import numpy as np



def leaky_relu(x, alpha):
    return LeakyReLU(alpha)(x)

def darknet_conv_BN_leaky(x, nb_filter, kernel_size,strides=(1,1), padding= 'same', name = None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name =None
        conv_name = None
    x = Conv2D(nb_filter,kernel_size, padding = padding, strides= strides)(x)
    x = BatchNormalization(name=bn_name)(x)
    x = leaky_relu(x,alpha=0.1)

    return x


def darknet_block(inpt, nb_filter,kernel_size=(1,1),strides=(1,1)):
    # the nb_filter is a list 
    x = darknet_conv_BN_leaky(inpt, nb_filter[0],kernel_size=(1,1), strides= (1,1), padding='same')
    x =darknet_conv_BN_leaky(x, nb_filter[1],kernel_size=(3,3),strides=(1,1),padding='same')
    x = add([x,inpt])
    return x

inpt  = Input(shape=(416,416,3))
x= darknet_conv_BN_leaky(inpt, nb_filter=32, kernel_size=(3,3))

x = ZeroPadding2D(((1,0),(1,0)))(x)
x = darknet_conv_BN_leaky(x,64,kernel_size=(3,3),strides=(2,2),padding='valid')
# 1 block
x = darknet_block(x, nb_filter=[32,64])  

x = ZeroPadding2D(((1,0),(1,0)))(x)
x = darknet_conv_BN_leaky(x,128,kernel_size=(3,3),strides=(2,2),padding='valid')
# 2 blocks
x = darknet_block(x,nb_filter=[64,128])
x = darknet_block(x, nb_filter=[64,128]) 

x = ZeroPadding2D(((1,0),(1,0)))(x)
x = darknet_conv_BN_leaky(x,256,kernel_size=(3,3),strides=(2,2),padding='valid')
# 8 blocks
x = darknet_block(x, nb_filter=[128,256])
x = darknet_block(x, nb_filter=[128,256])
x = darknet_block(x, nb_filter=[128,256])
x = darknet_block(x, nb_filter=[128,256])
x = darknet_block(x, nb_filter=[128,256])
x = darknet_block(x, nb_filter=[128,256])
x = darknet_block(x, nb_filter=[128,256])
x = darknet_block(x, nb_filter=[128,256])

x = ZeroPadding2D(((1,0),(1,0)))(x)
x = darknet_conv_BN_leaky(x,512,kernel_size=(3,3),strides=(2,2),padding='valid')
# 8 blocks
x= darknet_block(x, nb_filter=[256, 512])
x= darknet_block(x, nb_filter=[256, 512])
x= darknet_block(x, nb_filter=[256, 512])
x= darknet_block(x, nb_filter=[256, 512])
x= darknet_block(x, nb_filter=[256, 512])
x= darknet_block(x, nb_filter=[256, 512])
x= darknet_block(x, nb_filter=[256, 512])
x= darknet_block(x, nb_filter=[256, 512])

x = ZeroPadding2D(((1,0),(1,0)))(x)
x = darknet_conv_BN_leaky(x,1024,kernel_size=(3,3),strides=(2,2),padding='valid')
# 4 blocks
x= darknet_block(x, nb_filter=[512, 1024])
x= darknet_block(x, nb_filter=[512, 1024])
x= darknet_block(x, nb_filter=[512, 1024])
x= darknet_block(x, nb_filter=[512, 1024])
# this is the main body of darknet53
# next is the test of the model of darknet53 


x = AveragePooling2D()(x)
x= Flatten()(x)
x = Dense(1000,activation='softmax')(x)

model =Model(inputs=inpt,outputs=x)
sgd = SGD(decay=0.0001,momentum=0.9)  
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
# print the information of this  model  
model.summary()





    
    