# cifar100_incremental.py
# For RCL: create incremental CIFAR-100 dataset in pickle format

import numpy as np
import pickle
import argparse
import os
from tensorflow.keras.datasets import cifar100

parser = argparse.ArgumentParser()
default_path = os.path.join(os.path.dirname(__file__), 'cifar100_incremental.pkl')
parser.add_argument('--o', default=default_path, help='Output file path')
parser.add_argument('--n_tasks', default=10, type=int, help='Number of tasks to create')
parser.add_argument('--seed', default=42, type=int, help='Random seed')
args = parser.parse_args()

np.random.seed(args.seed)

# --- 1️⃣ Load CIFAR-100 dataset ---
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_train = y_train.flatten()
y_test = y_test.flatten()

# --- 2️⃣ Prepare incremental splits ---
num_classes = 100
classes_per_task = num_classes // args.n_tasks

tasks = []
for task_id in range(args.n_tasks):
    start = task_id * classes_per_task
    end = (task_id + 1) * classes_per_task
    task_classes = list(range(start, end))
    
    # Filter dataset for these classes
    train_mask = np.isin(y_train, task_classes)
    test_mask = np.isin(y_test, task_classes)
    
    x_tr_task = x_train[train_mask].reshape((-1, 32*32*3)) / 255.0
    y_tr_task = np.eye(num_classes)[y_train[train_mask]]  # one-hot
    
    x_te_task = x_test[test_mask].reshape((-1, 32*32*3)) / 255.0
    y_te_task = np.eye(num_classes)[y_test[test_mask]]  # one-hot
    
    tasks.append((x_tr_task, y_tr_task, x_te_task, y_te_task))
    print(f"Task {task_id+1}: classes {task_classes}")

# --- 3️⃣ Save to pickle file ---
output_dir = os.path.dirname(args.o)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
with open(args.o, 'wb') as f:
    pickle.dump(tasks, f)

print(f"\n✅ Saved CIFAR-100 incremental dataset with {args.n_tasks} tasks to {args.o}")
print(f"Absolute path: {os.path.abspath(args.o)}")
