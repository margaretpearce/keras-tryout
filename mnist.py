# Create a TensorFlow session and register it with Keras
import tensorflow as tf
from keras import backend as k
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data
from keras.metrics import categorical_accuracy as accuracy


sess = tf.Session()
k.set_session(sess)

# Placeholder to contain input digits
img = tf.placeholder(tf.float32, shape=(None, 784))

# Use Keras to define the model

# Fully connected layer with 128 units, ReLU activation
x = Dense(128, activation='relu')(img)
x = Dense(128, activation='relu')(x)

# Output layer with 10 units, softmax activation
preds = Dense(10, activation='softmax')(x)

# Define placeholder for the labels
labels = tf.placeholder(tf.float32, shape=(None, 10))

# Define the loss function
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# Train the model with TF optimizer
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize variables
k.get_session().run(tf.initialize_all_variables())

with sess.as_default():
    for i in range(5000):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0],
                                  labels: batch[1]})

# Evaluate the model
acc_values = accuracy(labels, preds)
with sess.as_default():
    print(acc_values.eval(feed_dict={img: mnist_data.test.images,
                                     labels: mnist_data.test.labels}))
