#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import rnn_cell

init_scale = 0.1
num_steps = 7
num_units = 7
input_data = [1, 2, 3, 4, 5, 6, 7]
target = [2, 3, 4, 5, 6, 7, 7]

batch_size = 1

with tf.Graph().as_default(), tf.Session() as session:
    # Placeholder for the inputs and target of the net
    # inputs = tf.placeholder(tf.int32, [batch_size, num_steps])
    input1 = tf.placeholder(tf.float32, [batch_size, 1])
    inputs = [input1 for _ in range(num_steps)]
    outputs = tf.placeholder(tf.float32, [batch_size, num_steps])

    gru = rnn_cell.GRUCell(num_units)
    initial_state = state = tf.zeros([batch_size, num_units])
    loss = tf.constant(0.0)

    # setup model: unroll
    for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        step_ = inputs[time_step]
        output, state = gru(step_, state)
        loss += tf.reduce_sum(abs(output - target))  # all norms work equally well? NO!
        final_state = state

    optimizer = tf.train.AdamOptimizer(0.1)  # CONVERGEs sooo much better
    train = optimizer.minimize(loss)  # let the optimizer train
    max_grad_norm=5
    trainable_variables = tf.trainable_variables()
    grads=tf.gradients(loss, trainable_variables)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    numpy_state = initial_state.eval()
    session.run(tf.initialize_all_variables())
    for epoch in range(10):  # now
        for i in range(7): # feed fake 2D matrix of 1 byte at a time ;)
            feed_dict = {initial_state: numpy_state, input1: [[input_data[i]]]} # no
            numpy_state, current_loss,_,_ = session.run([final_state, loss,train,train_op], feed_dict=feed_dict)
            print(current_loss)  # hopefully going down, always stuck at 189, why!?
