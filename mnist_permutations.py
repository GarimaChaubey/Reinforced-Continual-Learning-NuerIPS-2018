import tensorflow as tf
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--o', default='./mnist_permutations.pkl', help='output file')
parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
parser.add_argument('--seed', default=100, type=int, help='random seed')
args = parser.parse_args()
np.random.seed(args.seed)

# Load MNIST dataset using keras
(x_tr_raw, y_tr_raw), (x_te_raw, y_te_raw) = tf.keras.datasets.mnist.load_data()

# Preprocess data
x_tr_full = x_tr_raw.reshape(-1, 784).astype('float32') / 255.0
x_te = x_te_raw.reshape(-1, 784).astype('float32') / 255.0

# Convert labels to one-hot encoding
y_tr_full = tf.keras.utils.to_categorical(y_tr_raw, 10)
y_te = tf.keras.utils.to_categorical(y_te_raw, 10)

# Split training data into train and validation sets
val_size = 10000
x_tr = x_tr_full[:-val_size]
y_tr = y_tr_full[:-val_size]
x_val = x_tr_full[-val_size:]
y_val = y_tr_full[-val_size:]

permutations = []
for i in range(args.n_tasks):
    indices = np.random.permutation(784)
    permutations.append((x_tr[:, indices], y_tr, x_val[:, indices], y_val, x_te[:, indices], y_te))
f = open(args.o, "wb")
pickle.dump(permutations, f)
f.close()