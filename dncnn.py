from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from scipy import misc
import glob
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import os
def load_data():
    train_data=[]
    for i in range(15000):
        image_path = glob.glob("../train_images_64x64/train_%05d.png"%(i+4000))[0]
        image = misc.imread(image_path)
        train_data.append(image[:,:,0])
    train_data = np.array(train_data)

    true_data=[]
    for i in range(15000):
        image_path = glob.glob("../train_images_128x128/train_%05d.png"%(i+4000))[0]
        image = misc.imread(image_path)
        true_data.append(image[:,:,0])
    true_data = np.array(true_data)

    num_examples = len(true_data)
    print(num_examples)
    train_data = train_data.astype(float)
    true_data = true_data.astype(float)
    train_data = np.resize(train_data, (num_examples, 64,64,1))
    true_data = np.resize(true_data, (num_examples, 128,128,1))
    print(train_data.shape)
    print(true_data.shape)

    return train_data, true_data, num_examples

def load_data_val():
    train_data=[]
    for i in range(15000,16000):
        image_path = glob.glob("../train_images_64x64/train_%05d.png"%(i+4000))[0]
        image = misc.imread(image_path)
        train_data.append(image[:,:,0])
    train_data = np.array(train_data)

    true_data=[]
    for i in range(15000,16000):
        image_path = glob.glob("../train_images_128x128/train_%05d.png"%(i+4000))[0]
        image = misc.imread(image_path)
        true_data.append(image[:,:,0])
    true_data = np.array(true_data)

    num_examples = len(true_data)
    print(num_examples)
    train_data = train_data.astype(float)
    true_data = true_data.astype(float)
    train_data = np.resize(train_data, (num_examples, 64,64,1))
    true_data = np.resize(true_data, (num_examples, 128,128,1))
    print(train_data.shape)
    print(true_data.shape)

    return train_data, true_data

# Building the encoder
def model(x):
    out = tf.layers.conv2d(x, 64, 3, padding='same',activation=tf.nn.relu)
    
    for i in range(15):
        out = tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(out, 64, 3, padding='same')))
    
    out = tf.layers.conv2d_transpose(out, 1, 3, (2,2),padding='same')
    return out

def vae_train(train_data, true_data, num_examples):
    # Training Parameters
    learning_rate = 1e-5
    num_steps = 100000
    batch_size = 256

    display_step = 10

    val_x,val_y = load_data_val()

    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, 64, 64, 1])
    Y = tf.placeholder("float", [None, 128, 128, 1])

    # Construct model
    out = model(X)

    # Prediction
    y_pred = out
    # Targets (Labels) are the input data.
    y_true = Y

    # Define loss and optimizer, minimize the squared error
    loss = tf.losses.mean_squared_error(y_pred,y_true)
    #loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Start Training
    # Start a new TF session
    sess = tf.Session()

    # Run the initializer
    sess.run(init)

    # Training
    for s in range(1, num_steps+1):
    #     idx = np.random.choice(num_examples, batch_size)
    #     batch_x, batch_y = train_data[idx], true_data[idx]
        # Run optimization op (backprop) and cost op (to get loss value)
        
        #train_data, true_data = shuffle(train_data, true_data, random_state=s)
        idx = np.random.randint(15000, size=batch_size)
        batch_x, batch_y = train_data[idx], true_data[idx]
        # for offset in range(0, num_examples, batch_size):
            # end = offset + batch_size
            # batch_x, batch_y = train_data[offset:end], true_data[offset:end]
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, Y:batch_y})
        # Display logs per step
        if s % display_step == 0 or s == 1:
            print('Step %i: Minibatch Loss: %f learning_rate: %f' % (s, l, learning_rate))
        
        if s% 2000 ==0:
            learning_rate *= .8
        
        if s % 200 ==0:
            saver.save(sess, "./model/vae.ckpt")
            n = 1
            canvas_recon = np.empty((128 * n, 128 * n))
            for i in range(n):
                # MNIST test set
                batch_x = train_data[:batch_size]
                # Encode and decode the digit image
                g = sess.run(out, feed_dict={X: batch_x})
                
                # Display reconstructed images
                for j in range(n):
                    # Draw the generated digits
                    canvas_recon[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = g[j].reshape([128, 128])
            misc.imsave("output_%05d.png"%s, canvas_recon)

        if s%200 == 0:
            ct = sess.run([loss], feed_dict={X: val_x, Y:val_y})
            print("current validate loss is:",ct[0])
            saver.save(sess, "./model/vae_loss%d"%ct[0])

    

if __name__ == "__main__":
    train_data, true_data, num_examples = load_data()
    vae_train(train_data, true_data, num_examples)
    
