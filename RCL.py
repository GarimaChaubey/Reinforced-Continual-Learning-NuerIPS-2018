# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:27:34 2018

@author: Jason
"""
import warnings
warnings.filterwarnings("ignore")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
tf.compat.v1.keras.backend.clear_session()


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
from evaluate import evaluate
from policy_gradient import Controller
import argparse
import datetime
import time
import pickle
#from tqdm import tqdpPPm

class RCL:
    def __init__(self,args):
        self.args = args
        self.num_tasks = args.n_tasks
        self.epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.data_path = args.data_path
        self.max_trials = args.max_trials
        self.penalty = args.penalty
        self.task_list = self.create_mnist_task()
        self.evaluates = evaluate(task_list=self.task_list, args = args)
        self.train()

    def create_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        return sess

    def create_mnist_task(self):
        data = pickle.load(open(self.data_path, "rb"))
        # Detect input size from the data
        self.input_size = data[0][0].shape[1]  # Get feature dimension from first task's training data
        self.is_cifar = self.input_size == 3072
        if self.is_cifar:
            # For CIFAR: Much larger network for better feature extraction
            self.layer_sizes = [self.input_size, 2048, 1024, 100]
            # Convert CIFAR format (x_train, y_train, x_test, y_test) to MNIST format
            converted_data = []
            for task in data:
                x_train, y_train, x_test, y_test = task
                # MNIST format: x_train, y_train, x_val, y_val, x_test, y_test
                # For CIFAR we'll use test set as validation too
                converted_data.append((x_train, y_train, x_test, y_test, x_test, y_test))
            return converted_data
        else:
            # For MNIST: Original network (784 -> 312 -> 128 -> 10)
            self.layer_sizes = [self.input_size, 312, 128, 10]
            return data

    def train(self):
        self.best_params={}
        self.result_process = []
        for task_id in range(0,self.num_tasks):
            self.best_params[task_id] = [0,0]
            if task_id == 0:
                with tf.Graph().as_default() as g:
                    with tf.name_scope("before"):
                        '''inputs = tf.placeholder(shape=(None, 784), dtype=tf.float32)
                        y = tf.placeholder(shape=(None, 10), dtype=tf.float32)'''
                        
                        inputs = tf.compat.v1.placeholder(shape=(None, self.input_size), dtype=tf.float32)
                        output_size = 100 if self.is_cifar else 10
                        y = tf.compat.v1.placeholder(shape=(None, output_size), dtype=tf.float32)

                        # Use the detected layer sizes
                        w1 = tf.Variable(tf.random.truncated_normal(shape=(self.layer_sizes[0], self.layer_sizes[1]), stddev=0.01))
                        b1 = tf.Variable(tf.constant(0.1, shape=(self.layer_sizes[1],)))
                        w2 = tf.Variable(tf.random.truncated_normal(shape=(self.layer_sizes[1], self.layer_sizes[2]), stddev=0.01))
                        b2 = tf.Variable(tf.constant(0.1, shape=(self.layer_sizes[2],)))
                        w3 = tf.Variable(tf.random.truncated_normal(shape=(self.layer_sizes[2], self.layer_sizes[3]), stddev=0.01))
                        b3 = tf.Variable(tf.constant(0.1, shape=(self.layer_sizes[3],)))
                        '''output1 = tf.nn.relu(tf.nn.xw_plus_b(inputs,w1,b1,name="output1"))
                        output2 = tf.nn.relu(tf.nn.xw_plus_b(output1,w2,b2,name="output2"))
                        output3 = tf.nn.xw_plus_b(output2,w3,b3,name="output3")'''
                        
                        # Input normalization and dropout
                        inputs_normalized = tf.nn.l2_normalize(inputs, axis=1)
                        inputs_dropped = tf.nn.dropout(inputs_normalized, rate=0.2)
                        
                        # First layer with dropout
                        h1 = tf.matmul(inputs_dropped, w1) + b1
                        output1 = tf.nn.relu(h1)
                        output1 = tf.nn.dropout(output1, rate=0.3)
                        
                        # Second layer with dropout
                        h2 = tf.matmul(output1, w2) + b2
                        output2 = tf.nn.relu(h2)
                        output2 = tf.nn.dropout(output2, rate=0.3)
                        
                        # Output layer
                        output3 = tf.matmul(output2, w3) + b3


                        # Adjust regularization for CIFAR
                        reg_factor = 0.00001 if self.is_cifar else 0.0001
                        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output3)) + \
                               reg_factor*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))
                        
                        # Choose and configure optimizer
                        if self.args.optimizer=="adam":
                            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.args.lr)
                        elif self.args.optimizer=="rmsprop":
                            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.lr)
                        elif self.args.optimizer=="sgd":
                            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.lr)
                        else:
                            raise Exception("please choose one optimizer")
                        train_step = optimizer.minimize(loss)
                        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1),tf.argmax(output3,axis=1)),tf.float32))
                        
                        '''sess = self.create_session()
                        sess.run(tf.global_variables_initializer())'''

                        tf.compat.v1.disable_eager_execution()
                        sess = tf.compat.v1.Session()
                        sess.run(tf.compat.v1.global_variables_initializer())
                        tf.compat.v1.get_default_graph().finalize()

                        train_data = self.task_list[task_id][0]
                        train_labels = self.task_list[task_id][1]
                        n_samples = len(train_data)
                        
                        for epoch in range(self.epochs):
                            # Shuffle the training data
                            perm = np.random.permutation(n_samples)
                            train_data_shuffled = train_data[perm]
                            train_labels_shuffled = train_labels[perm]
                            
                            total_loss = 0
                            num_batches = 0
                            
                            for start_idx in range(0, n_samples, self.batch_size):
                                end_idx = min(start_idx + self.batch_size, n_samples)
                                batch_xs = train_data_shuffled[start_idx:end_idx]
                                batch_ys = train_labels_shuffled[start_idx:end_idx]
                                
                                _, batch_loss = sess.run([train_step, loss], feed_dict={inputs:batch_xs, y:batch_ys})
                                total_loss += batch_loss
                                num_batches += 1
                            if epoch % 2 == 0:
                                avg_loss = total_loss / num_batches
                                print(f"Task {task_id}, Epoch {epoch}: avg_loss = {avg_loss:.4f}")
                        accuracy_test = sess.run(accuracy, feed_dict={inputs:self.task_list[task_id][4], y:self.task_list[task_id][5]})
                        print(f"task:{task_id},test accuracy:{accuracy_test}")
                        self.vars = sess.run([w1,b1,w2,b2,w3,b3])
                    self.best_params[task_id] = [accuracy_test,self.vars]
            else:
                tf.compat.v1.reset_default_graph()
                controller = Controller(self.args)
                results = []
                best_reward = 0
                for trial in range(self.max_trials):
                    actions = controller.get_actions()
                    print("***************actions*************",actions)
                    accuracy_val, accuracy_test = self.evaluates.evaluate_action(var_list = self.vars, 
                             actions=actions, task_id = task_id)

                    results.append(accuracy_val)
                    print("test accuracy: ", accuracy_test)
                    reward = accuracy_val - self.penalty*sum(actions)
                    print("reward: ", reward)
                    if reward > best_reward:
                        best_reward = reward
                        self.best_params[task_id] = (accuracy_test, self.evaluates.var_list)
                    controller.train_controller(reward)
                controller.close_session()
                self.result_process.append(results)
                self.vars = self.best_params[task_id][1]
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reinforced Continual learning')

    # model parameters
    parser.add_argument('--n_tasks', type=int, default=10,
                        help='number of tasks')
    parser.add_argument('--n_hiddens', type=str, default='312,218',
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')

    # optimizer parameters
    parser.add_argument('--n_epochs', type=int, default=15,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='SGD learning rate')
    parser.add_argument('--max_trials', type=int, default=50,
                        help='max_trials')

    # experiment parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    parser.add_argument('--save_path', type=str, default='./results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--data_path', default='./data/mnist_permutations.pkl',
                        help='path where data is located')
    parser.add_argument('--state_space', type=int, default=30, help="the state space for search") 
    parser.add_argument('--actions_num', type=int, default=2, help="how many actions to dscide")
    parser.add_argument('--hidden_size', type=int, default=100, help="the hidden size of RNN")
    parser.add_argument('--num_layers', type=int, default=2, help="the layer of a RNN cell")
    parser.add_argument('--cuda', type=bool, default=True, help="use GPU or not")
    parser.add_argument('--bendmark', type=str, default='critic', help="the type of bendmark")
    parser.add_argument('--penalty', type=float, default=0.0001, help="the type of bendmark")#0.0001
    parser.add_argument('--optimizer', type=str, default="adam", help="the type of optimizer")#
    parser.add_argument('--method', type=str, default='policy', help="method for generate actions")

    args = parser.parse_args()
    start = time.time()
    jason = RCL(args)  
    end = time.time()
    params = jason.best_params
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # Use a filesystem-safe timestamp (avoid ':' which is invalid on Windows filenames)
    fname = "RCL_FC_" + args.data_path.split('/')[-1] + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname += '_' + str(args.lr) + str("_") + str(args.n_epochs) + '_' + str(args.max_trials) + '_' + str(args.batch_size) + \
             '_' + args.bendmark + '_' + str(args.penalty) + '_' + args.optimizer + '_' + str(args.state_space) + '_' + \
             str(end-start) + '_' + args.method
    fname = os.path.join(args.save_path, fname)
    f = open(fname + '.txt', 'w')
    accuracy = []
    for index,value in params.items():
        print([_.shape for _ in value[1]], file=f)
        accuracy.append(value[0])
    print("FINAL ACCURACIES:", accuracy, file=f)
    print("\nDETAILED SHAPES:", file=f)
    for index,value in params.items():
        print(f"Task {index} shapes:", [_.shape for _ in value[1]], file=f)
    f.close()
    print(fname)
    name = fname + '.pkl'
    f = open(name, 'wb')
    pickle.dump(jason.result_process, f)
    f.close()
