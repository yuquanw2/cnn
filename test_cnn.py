from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from scipy import misc
import glob
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

train_data=[]
for i in range(1,4000):
    image_path = glob.glob("../test_images_64x64/test_%05d.png"%(i))[0]
    image = misc.imread(image_path)
    train_data.append(image[:,:,0])
train_data = np.array(train_data)

num_examples = len(train_data)
print(num_examples)
train_data = train_data.astype(float)
train_data = np.resize(train_data, (num_examples, 64,64,1))
print(train_data.shape)

# Building the encoder
def model(x):
    out = tf.layers.conv2d(x, 64, 3, padding='same',activation=tf.nn.relu)
    
    for i in range(15):
        out = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(out, 64, 3, padding='same')))
    
    out = tf.layers.conv2d_transpose(out, 1, 3, (2,2),padding='same')
    return out

X = tf.placeholder("float", [None, 64, 64, 1])
# Construct model
decoder_op = model(X)

saver = tf.train.Saver()
sess = tf.Session()
path = tf.train.latest_checkpoint('./model/')
saver.restore(sess, path)

batch_size = 256


# MNIST test set
#batch_x = train_data[:batch_size]
# Encode and decode the digit image
for offset in range(0, num_examples, batch_size):
    end = offset + batch_size
    g = sess.run(decoder_op, feed_dict={X: train_data[offset:end]})
    for i in range(len(g)): 
        canvas_recon = g[i].reshape([128, 128])
        print("test_%05d.png"%(offset+i+1))
        misc.imsave("./test/test_%05d.png"%(offset+i+1), canvas_recon)


