import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and
    softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout (which you need to
        # implement!!!!)
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            print 'W.get_shape()', W.get_shape()
            print 'self.input_x.get_shape()', self.input_x.get_shape()
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            print 'self.embedded_chars', self.embedded_chars.get_shape()
            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)
            print 'self.embedded_chars_expanded', self.embedded_chars_expanded.get_shape()

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # conv1
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # define filte shape
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                print 'filter_shape', filter_shape
                # define kernel/filter matrix; filter shape is outputed
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                # define biases
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                # https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#conv2d
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print 'conv', conv
                # Add bias to convolution
                bias = tf.nn.bias_add(conv, b)
                print 'bias', bias
                # Apply nonlinearity
                # h = tf.nn.relu(bias, name="reclu")
                h = tf.nn.tanh(bias, name="tanh")
                print 'h', h

            # Maxpooling over the outputs
            # https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html#max_pool
            print 'sequence_length', sequence_length
            print 'filter_size', filter_size
            pool1 = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool1")
            print 'pool1', pool1
            pooled_outputs.append(pool1)
            print 'len(pooled_outputs)', len(pooled_outputs)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        print 'num_filters_total', num_filters_total
        self.h_pool = tf.concat(3, pooled_outputs)
        print 'self.h_pool', self.h_pool
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Reshaping when we need to use two CNNs
        # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total, 1, 1])
        # print 'self.h_pool_flat', self.h_pool_flat

        # pooled_outputs = []
        # for i, filter_size in enumerate(filter_sizes):
        #     with tf.name_scope("conv-maxpool-2-%s" % filter_size):
        #         # Convolution Layer
        #         filter_shape = [filter_size, 1, 1, num_filters]
        #         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        #         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        #         conv = tf.nn.conv2d(
        #             self.h_pool_flat,
        #             W,
        #             strides=[1, 1, 1, 1],
        #             padding="VALID",
        #             name="conv")
        #         # Apply nonlinearity
        #         # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #         h = tf.nn.tanh(tf.nn.bias_add(conv, b), name="tanh")
        #         # print 'h convo 2 h.get_shape()', h.get_shape()[1]

        #         # Maxpooling over the outputs
        #         pooled = tf.nn.max_pool(
        #             h,
        #             ksize=[1, h.get_shape()[1], 1, 1],
        #             strides=[1, 1, 1, 1],
        #             padding='VALID',
        #             name="pool")
        #         print 'pooled +++ ', pooled
        #         pooled_outputs.append(pooled)

        # Maxpooling over the outputs for two CNN
        # self.h_pool_2 = tf.concat(3, pooled_outputs)
        # self.h_pool_flat_2 = tf.reshape(self.h_pool_2, [-1, num_filters_total])
        # print 'self.h_pool_flat_2', self.h_pool_flat_2

        # Maxpooling for one CNN
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print 'self.h_pool_flat', self.h_pool_flat

        # Add dropout
        # ##############add your code here################
        # hint: you need to add dropout on self.h_pool_flat with
        # Added by MATT DUNN 9/21/16
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            print 'W.get_shape()', W.get_shape()
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                self.h_drop, W, b, name="scores")
            print 'self.scores', self.scores
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
