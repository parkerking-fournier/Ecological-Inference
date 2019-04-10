from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Training and test sets
X_TRAINING = "../../../datasets/npy/train/x/train_x.npy"
Y_TRAINING = "../../../datasets/npy/train/y/classification/train_y.npy"
X_TEST = "../../../datasets/npy/test/x/test_x.npy"
Y_TEST = "../../../datasets/npy/test/y/classification/test_y.npy"

# Load files
print("Loading files...")
train_x = np.load(X_TRAINING)
train_y = np.load(Y_TRAINING)
train_y = train_y.astype(int)
test_x  = np.load(X_TEST)
test_y  = np.load(Y_TEST)
test_y = test_y.astype(int)
print("     Finished!")

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[149769])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[50, 50, 50, 50, 50],
                                        n_classes=3,
                                        model_dir="/Users/parkerkingfournier/Documents/University/U4(2017-18)/Summer/ResearchProject/python/scripts/models/logs/classifiers/dnn_classifier",
                                        dropout=0.5)

# Define the training and eval inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": train_x},
    y = train_y,
    batch_size = 50,
    num_epochs = None,
    shuffle = True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": test_x},
    y = test_y,
    num_epochs = 1,
    shuffle = False)

# Train model.
print("\nTraining model...")
classifier.train(input_fn=train_input_fn, steps=200)
print("     Finished!")

# Evaluate accuracy.
print("\nTesting model...")
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("     Finished!")

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))