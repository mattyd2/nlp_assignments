import tensorflow as tf
import numpy as np


class cbow(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and 
    softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, hidden_size, l2_reg_lambda=0.0):

        # input placeholders
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
            self.embedded_words = tf.nn.embedding_lookup(W, self.input_x)
            self.mean = tf.reduce_mean(self.embedded_words, 1)

        # hidden layer
        with tf.name_scope("hidden_layer_1"):
            W = tf.Variable(tf.truncated_normal(
                [embedding_size, hidden_size], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b")
            self.scores_hidden = tf.nn.xw_plus_b(
                self.mean, W, b, name="scores")
            self.hidden_1 = tf.nn.relu(self.scores_hidden, name="relu")

        # dropout
        with tf.name_scope("dropout_1"):
            self.h_drop_1 = tf.nn.dropout(
                self.hidden_1, self.dropout_keep_prob)

        # hidden layer #2
        with tf.name_scope("hidden_layer_2"):
            W = tf.Variable(tf.truncated_normal(
                [embedding_size, hidden_size], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="b")
            self.scores_hidden_2 = tf.nn.xw_plus_b(
                self.h_drop_1, W, b, name="scores")
            self.hidden_2 = tf.nn.relu(self.scores_hidden_2, name="relu")

        # dropout #2
        with tf.name_scope("dropout_2"):
            self.h_drop_2 = tf.nn.dropout(
                self.hidden_2, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[hidden_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop_2, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")
