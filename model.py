# -*- coding: utf-8 -*-

import tensorflow as tf


class Model:
    def __init__(self,
                 is_train,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9995):
        self.x_ = tf.placeholder(tf.float32, [None, 28 * 28])
        self.y_ = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        self.W1 = tf.Variable(weight_variable([28 * 28, 100]), trainable=True)
        self.b1 = tf.Variable(bias_variable([100]), trainable=True)
        self.linear1 = tf.matmul(self.x_, self.W1) + self.b1

        self.BN = batch_normalization_layer(self.linear1, isTrain=is_train)

        self.Relu = tf.nn.relu(self.BN)

        self.W2 = tf.Variable(weight_variable([100, 10]), trainable=True)
        self.b2 = tf.Variable(bias_variable([10]), trainable=True)
        self.linear2 = tf.matmul(self.Relu, self.W2) + self.b2

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_, logits=self.linear2))
        self.correct_pred = tf.equal(tf.cast(tf.argmax(self.linear2, 1), tf.int32), self.y_)
        self.pred = tf.argmax(self.linear2, 1)  # Calculate the prediction result
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))  # Calculate the accuracy in this mini-batch

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)  # Learning rate decay

        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step,
                                                                            var_list=self.params)  # Use Adam Optimizer

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2,
                                    max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)


def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def batch_normalization_layer(inputs, isTrain):
    gamma = tf.Variable(tf.ones([100]), trainable=True, dtype=tf.float32)
    beta = tf.Variable(tf.zeros([100]), trainable=True, dtype=tf.float32)

    size = tf.Variable(tf.constant(0.0), trainable=False, dtype=tf.float32)
    total_mean = tf.Variable(tf.zeros([100]), trainable=False, dtype=tf.float32)
    total_variance = tf.Variable(tf.zeros([100]), trainable=False, dtype=tf.float32)

    pop_mean = tf.Variable(tf.zeros([100]), trainable=False, dtype=tf.float32)
    pop_variance = tf.Variable(tf.ones([100]), trainable=False, dtype=tf.float32)

    epsilon = tf.constant(1e-5)

    if isTrain:

        mean, variance = tf.nn.moments(inputs, [0])

        total_mean = tf.add(total_mean, mean)
        total_variance = tf.add(total_variance, variance)

        size = tf.add(size, 1.0)

        est_mean = tf.assign(pop_mean, tf.divide(total_mean, size))
        est_variance = tf.assign(pop_variance, tf.divide(total_variance, size))
        with tf.control_dependencies([est_mean, est_variance]):
            return tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_variance, beta, gamma, epsilon)
