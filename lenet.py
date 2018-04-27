from scipy import sparse, io
import numpy as np
import tensorflow as tf

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def conv_m(filename, h, w, d, ho, wo, do, size):
    a = io.mmread(filename)
    after = set()
    index = sparse.find(a)
    x = index[0]
    y = index[1]
    for i in range(len(x)):
        after.add((x[i], y[i]))
    weight = set()
    for i in after:
        if (i[1],i[0]) not in weight and i[0]!=i[1]:
            weight.add(i)

    out = np.zeros((do, ho, wo, d*size**2))
    for i in weight:
        if i[0]>=w*h*d:
            s = i[1]
            l = i[0]
        else:
            s = i[0]
            l = i[1]
        in_d = s//(h*w)
        in_h = s%(h*w)//w
        in_w = s%(h*w)%w
        o = l-w*h*d
        out_h = o%(wo*ho)//wo
        out_w = o%(wo*ho)%wo
        out_d = o//(wo*ho)
        weight_h = in_h - out_h
        weight_w = in_w - out_w
        out[out_d][out_h][out_w][size**2*in_d+size*weight_h+weight_w] = 1.
   # for ddo in range(do):
    #    for dd in range(d):
     #       for i in range(size):
      #          for j in range(size):
                    # if conv[i][j][dd][ddo] >= .5:
       #             out[ddo,:,:,size**2*dd+size*i + j] = 1.
    out = tf.constant(out, dtype=tf.float32)
    return out

def fc_m(filename, in_n, out_n):
    n = in_n + out_n
    a = io.mmread(filename)
    after = set()
    index = sparse.find(a)
    x = index[0]
    y = index[1]
    for i in range(len(x)):
        after.add((x[i], y[i]))
    weight = set()
    for i in after:
        if (i[1],i[0]) not in weight and i[0]!=i[1]:
            weight.add(i)

    fc = set()
    for i in weight:
        if i[0]< n and i[1]<n and i[0]>=0 and i[1]>=0:
            x = i[0]
            y = i[1]
            if x>=in_n:
                fc.add((y,x-in_n))
            else:
                fc.add((x,y-in_n))

    out = np.zeros((in_n,out_n))

    for i in list(fc):
        out[i[0],i[1]] = 1
    out = tf.constant(out, dtype=tf.float32)
    return out

def conv_2d(x, w, m):
    w_flat = tf.reshape(w, [-1,int(w.shape[3])])
    x_patch = tf.extract_image_patches(x, 
                                 ksizes = [1, w.shape[0], w.shape[1],1],
                                 strides = [1,1,1,1],
                                 rates = [1,1,1,1],
                                 padding = 'VALID')
    feature_maps = []
    for i in range(w.shape[3]):
        patch = tf.multiply(x_patch, m[i,:,:,:])
        feature_map = tf.reduce_sum(tf.multiply(w_flat[:, i], patch), axis=3, keep_dims=True)
        feature_maps.append(feature_map)
    features = tf.concat(feature_maps, axis=3)  
    return features

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

import random
import numpy as np

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

EPOCHS = 200
BATCH_SIZE = 8

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
#     (feature, weight,n, ow, oh, od, conv_m):
#     conv1 = conv_mul(x, conv1_w, 8, 28, 28, 6, conv_m )
    conv1_b = tf.Variable(tf.zeros(6))
    
    m = conv_m('lenet_conv1_b10.mtx', 32, 32, 1, 28, 28, 6, 5)
    conv1 = conv_2d(x, conv1_w, m) + conv1_b
    
    #conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
#     conv1 = tf.multiply(conv1, pool_m1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    m = conv_m('lenet_conv2_b10.mtx', 14, 14, 6, 10, 10, 16, 5)
    conv2 = conv_2d(pool_1, conv2_w, m) + conv2_b
    #     conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
#     conv2 = tf.multiply(conv2, pool_m2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_m = fc_m('lenet_fc1_b10.mtx', 400,120)
    fc1_w = tf.multiply(fc1_w, fc1_m)
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_m = fc_m('lenet_fc2_b10.mtx', 120,84)
    fc2_w = tf.multiply(fc2_w, fc2_m)
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
    fc3_m = fc_m('lenet_fc3_b10.mtx', 84,10)
    fc3_w = tf.multiply(fc3_w, fc3_m)
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf. train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

#     conv1_weight = conv1_w.eval()
#     conv2_weight = conv2_w.eval()
#     fc1_weight = fc1_w.eval()
#     fc2_weight = fc2_w.eval()
#     fc3_weight = fc3_w.eval()
#     np.save('conv1_weight', conv1_weight)
#     np.save('conv2_weight', conv2_weight)
#     np.save('fc1_weight', fc1_weight)
#     np.save('fc2_weight', fc2_weight)
#     np.save('fc3_weight', fc3_weight)

    print("Complete")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
