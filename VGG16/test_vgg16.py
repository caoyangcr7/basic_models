import tensorflow as tf
import numpy as np
import VGG_16
import utils


im1 = utils.load_image('./cat.jpeg')
im2 = utils.load_image('./tiger.jpeg')
print(im1.shape,'image_shape')

batch1 = im1.reshape((1,224,224,3))
batch2 =im2.reshape((1,224,224,3))
batch = np.concatenate((batch1, batch2), 0)


with tf.Session() as sess:
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = VGG_16.VGG_16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    # print(type(prob))
    # print(vgg.conv3_1)
    # print(sess.run(vgg.conv3_1),feed_dict=feed_dict)
    # conv_3_1 = sess.run(vgg.conv3_1,feed_dict=feed_dict)
    # print (conv_3_1)
    # print (np.argmax(prob))
    # print(prob.shape)
    
    utils.print_prob(prob[0], './synset.txt')
    utils.print_prob(prob[1], './synset.txt')