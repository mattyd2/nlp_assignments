import numpy as np
import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print result

# tns = tf.Tensor()
# import numpy
# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
rng = np.random


# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

# Training Data
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779,
                         6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366,
                         2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

# plt.plot(train_X, train_Y, 'ro', label='Original data')

# tf Graph Input (X, y)
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tip: initialize X and y as placeholders of float
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create Model
# Set model weights (W, b)
# tip: initialize W and b as randome variable with value rng.randn()
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
# activation = W * X + b
pred = tf.add(tf.mul(X, W), b)

# Define the cost function (the squared errors)
# cost = (activation - y)**2 / (2 * n_sample)
# use tf.train.GradientDescentOptimizer() as optimizer

cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                  "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=",
          sess.run(W), "b=", sess.run(b), '\n')

