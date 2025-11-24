# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 18:45:08 2018

@author: Jason
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class PolicyEstimator:
    """
    Policy Function approximator (policy network using an LSTM controller).
    This implementation uses Keras LSTM layers but runs inside the TF1 graph
    (eager execution is disabled in the workspace). The controller emits a
    sequence of actions; at each step the LSTM output is mapped to a
    probability distribution over the state space.
    """

    def __init__(self, args, learning_rate=0.01, scope="policy_estimator"):
        self.args = args
        self.input_size = args.state_space
        self.state_space = args.state_space
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.actions_num = args.actions_num

        # Placeholders
        self.state = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.state_space), name="states")
        self.actions = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.actions_num, ), name="actions")
        self.target = tf.compat.v1.placeholder(dtype=tf.float32, name="target")

        # Output projection: from LSTM hidden state -> logits over state_space
        self.hidden2output_w = tf.Variable(tf.random.truncated_normal(shape=(self.hidden_size, self.state_space), stddev=0.01))
        self.hidden2output_b = tf.Variable(tf.constant(0.1, shape=(self.state_space,)))

        # Build stacked LSTM layers using Keras (works when eager is disabled)
        lstm_layers = []
        for _ in range(self.num_layers):
            # We process one timestep at a time, so return_sequences=False
            lstm = tf.keras.layers.LSTM(self.hidden_size, return_sequences=False, return_state=True)
            lstm_layers.append(lstm)

        # We run with batch_size=1 in the existing code (controller uses a single state)
        batch_size = 1
        # initialize h,c for each layer
        current_states = []
        for _ in range(self.num_layers):
            h0 = tf.zeros([batch_size, self.hidden_size])
            c0 = tf.zeros([batch_size, self.hidden_size])
            current_states.append([h0, c0])

        inputs = self.state
        self.outputs = []

        # Process the sequence of decisions (actions_num timesteps)
        for _ in range(self.actions_num):
            # reshape inputs to [batch, time, features] where time=1
            current_input = tf.reshape(inputs, [batch_size, self.state_space])
            current_input = tf.expand_dims(current_input, axis=1)

            lstm_input = current_input
            new_states = []
            # pass through stacked LSTM layers
            for layer_idx, lstm in enumerate(lstm_layers):
                # lstm returns (output, state_h, state_c)
                lstm_out, state_h, state_c = lstm(lstm_input, initial_state=current_states[layer_idx])
                # lstm_out shape: [batch, units]
                # prepare input for next layer: expand dims to [batch, time=1, features]
                lstm_input = tf.expand_dims(lstm_out, axis=1)
                new_states.append([state_h, state_c])

            current_states = new_states
            # final output from last layer is lstm_out
            cell_output = lstm_out

            # map to probabilities over state_space
            logits = tf.matmul(cell_output, self.hidden2output_w) + self.hidden2output_b
            output = tf.nn.softmax(logits)
            self.outputs.append(output)

            # next input is the previous output (works as in original implementation)
            inputs = output

        # Compute product of picked action probabilities across timesteps
        picked_action_prob = None
        for t in range(self.actions_num):
            # outputs[t] shape: [batch, state_space]
            prob_t = self.outputs[t][0, self.actions[t]]
            if t == 0:
                picked_action_prob = prob_t
            else:
                picked_action_prob = picked_action_prob * prob_t

        # Loss and training op
        self.loss = -tf.math.log(picked_action_prob + 1e-12) * self.target
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.args.lr)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess if sess else tf.get_default_session()
        return sess.run(self.outputs, {self.state: state})

    def update(self, state, target, actions, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target, self.actions: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator:
    """
    Value Function approximator.
    """

    def __init__(self, args, learning_rate=0.005, scope="value_estimator"):
        self.state_space = args.state_space
        self.state = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, self.state_space), name="states")
        self.target = tf.compat.v1.placeholder(dtype=tf.float32, name="target")

        # This is just table lookup estimator
        self.state_reshaped = tf.reshape(self.state, shape=(1, self.state_space))
        
        # Create weights and biases for dense layer
        self.dense_w = tf.Variable(tf.zeros([self.state_space, 1]))
        self.dense_b = tf.Variable(tf.zeros([1]))
        self.output_layer = tf.matmul(self.state_reshaped, self.dense_w) + self.dense_b
        self.value_estimate = tf.squeeze(self.output_layer)
        self.loss = tf.compat.v1.losses.mean_squared_error(self.target, self.value_estimate)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess if sess else tf.get_default_session()
        return sess.run(self.value_estimate, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class Controller:
    def __init__(self, args, scope="Controller"):
        self.args = args
        self.state = np.random.random(size=(1, args.state_space))
        self.policy_estimator = PolicyEstimator(args)
        self.value_estimator = ValueEstimator(args)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train_controller(self, reward):
        baseline_value = self.value_estimator.predict(self.state, self.sess)
        advantage = reward - baseline_value
        self.value_estimator.update(self.state, reward, self.sess)
        self.policy_estimator.update(self.state, advantage, self.actions, self.sess)

    def get_actions(self):
        action_probs = self.policy_estimator.predict(self.state, self.sess)
        self.actions = []
        for i in range(self.args.actions_num):
            prob = action_probs[i]
            action = np.random.choice(np.arange(self.args.state_space),p=prob[0])
            self.actions.append(action)
        return self.actions

    def close_session(self):
        self.sess.close()
