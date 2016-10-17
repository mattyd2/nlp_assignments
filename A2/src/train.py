#! /usr/bin/env python
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers as dh
# from sklearn.cross_validation import train_test_split
from text_cbow import cbow
from tensorflow.contrib import learn
# import time
# import pickle

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128,
                        "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("max_sentence_length", 100,
                        "The maximum sentence length.")
tf.flags.DEFINE_integer("hidden_layer_size", 1000,
                        "Dimension of the hidden layer")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 1,
                      "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("N_grams", 3,
                      "Number of n_grams")

# Training parameters
tf.flags.DEFINE_integer("patience_wait", 600,
                        "Steps to wait while performance decreases (default: 500)")

# TODO: change to 64
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer(
    "num_epochs", 50, "Number of training epochs (default: 200)")

# add by MD - to be used for triggering validation checks
tf.flags.DEFINE_integer("validate_every", 200,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 200,
                        "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200,
                        "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# # create the merge files if they don't already exist
dh.create_merge_files()

# # get the prepared data
# start = time.time()
# x_train, y_train, x_val, y_val, x_test, y_test, vocab_processor = dh.get_prepared_data(
#     FLAGS.max_sentence_length, FLAGS.N_grams)
# end = time.time()
# print('Time to Process Data...')
# print(end - start)

# to_pickle = [x_train, y_train, x_val, y_val, x_test, y_test, vocab_processor]
labels = ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test', 'vocab_processor']
# dh.pickle_data(to_pickle, labels, FLAGS.N_grams)

start = time.time()
x_train, y_train, x_val, y_val, x_test, y_test, vocab_processor = dh.get_pickled_data(labels, FLAGS.N_grams)
end = time.time()
print('Time to Retrieve Pickles...')
print(end - start)

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # initialize the cbow model
        cbow_ = cbow(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            hidden_size=FLAGS.hidden_layer_size,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # track every step the optimizer takes
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # define starting learningn rate
        starter_learning_rate = 1e-2

        # reduce the learning rate as the model is trained
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   200, 0.96, staircase=True)
        # call optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # compute the gradients using the loss defined in text_cbow
        grads_and_vars = optimizer.compute_gradients(cbow_.loss)

        # apply the gradients to the model
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(
            os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cbow_.loss)
        acc_summary = tf.scalar_summary("accuracy", cbow_.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(
            train_summary_dir, sess.graph)

        # Validation Summaries
        val_summary_op = tf.merge_summary(
            [loss_summary, acc_summary, grad_summaries_merged])
        val_summary_dir = os.path.join(out_dir, "summaries", "validation")
        val_summary_writer = tf.train.SummaryWriter(
            val_summary_dir, sess.graph)

        # test summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(
            dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already
        # exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cbow_.input_x: x_batch,
                cbow_.input_y: y_batch,
                # Matt Dunn add 9/21
                cbow_.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op,
                    cbow_.loss, cbow_.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(
            #     time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def val_step(x_batch, y_batch):
            """
            Eval model on validation set
            """
            feed_dict = {
                cbow_.input_x: x_batch,
                cbow_.input_y: y_batch,
                # Matt Dunn add 9/21
                cbow_.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cbow_.loss, cbow_.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))
            val_summary_writer.add_summary(summaries, step)
            return accuracy

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cbow_.input_x: x_batch,
                cbow_.input_y: y_batch,
                # Added Matt Dunn 9/21
                cbow_.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cbow_.loss, cbow_.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(
                time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = dh.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        step_for_max_val_accuracy = 0
        max_val_accuracy = 0.0
        previous_val_accuracy = 0.0

        # run session
        start = time.time()
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.validate_every == 0:
                print("\nValidation Evaluation:")
                new_val_accuracy = val_step(x_val, y_val)
                print("")
                path = saver.save(sess, checkpoint_prefix,
                                  global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

                # Case: New Val Accuracy is Greater Than or Equal to Previous
                # print 'new_val_accuracy', new_val_accuracy
                # print 'previous_val_accuracy', previous_val_accuracy

                if (new_val_accuracy - 0.01) > previous_val_accuracy:
                    # print 'Case: New Val Accuracy is Greater Than or Equal to
                    # Previous'
                    step_for_max_val_accuracy = current_step
                    max_val_accuracy = new_val_accuracy

                # Case: New Val Accuracy is Less Than Previous
                else:
                    # print 'Case: New Val Accuracy is Less Than Previous'
                    steps_taken = current_step - step_for_max_val_accuracy
                    # print 'steps_taken', steps_taken
                    # Patience Level Exceeded, eval test set
                    if steps_taken > FLAGS.patience_wait:
                        print("\nTest Set Evaluation:")
                        dev_step(x_test, y_test, writer=dev_summary_writer)
                        print("")
                        end = time.time()
                        print('Time to Process Data...')
                        print(end - start)
                        break
                # print 'previous_val_accuracy = new_val_accuracy'
                previous_val_accuracy = new_val_accuracy
                # print 'previous_val_accuracy', previous_val_accuracy
