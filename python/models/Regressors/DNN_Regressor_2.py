from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import json

# Training and test sets
X_TRAINING = "../../../datasets/npy/train/x/train_x.npy"
Y_TRAINING = "../../../datasets/npy/train/y/regression/train_y_characteristic_length_normalized.npy"
X_TEST = "../../../datasets/npy/test/x/test_x.npy"
Y_TEST= "../../../datasets/npy/test/y/regression/test_y_characteristic_length_normalized.npy"

# Declare and open results file
RESULTS_FILE = "/Users/parkerkingfournier/Documents/Development/workspace/projects/Ecological-Inference/python/models/Regressors/logs/dnn_characteristic_length_2_normalized/results.txt"
file = open(RESULTS_FILE, "w+")

# Load files
print("\nLoading files...")
train_x = np.load(X_TRAINING)
train_y = np.load(Y_TRAINING)
test_x  = np.load(X_TEST)
test_y  = np.load(Y_TEST)
print("     Finished!")

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[149769])]

# Build 5 layer DNN with 50, 50, 50, 50, 50 units respectively.
classifier = tf.estimator.DNNRegressor( feature_columns=feature_columns,
                                        hidden_units=[50, 50, 50, 50, 50],
                                        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000000000001),
                                        model_dir="/Users/parkerkingfournier/Documents/Development/workspace/projects/Ecological-Inference/python/models/Regressors/logs/dnn_characteristic_length_2_normalized",
                                        dropout=0.5)



# Define the training and eval inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_x},
    y=train_y,
    batch_size=50,
    num_epochs=None,
    shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_x},
    y=test_y,
    num_epochs=1,
    shuffle=False)

# Training related variables
training_steps      = 25
training_sessions   = 40

for training_session in xrange(0,training_sessions):
    # Train model.
    print("\nTraining model at step ", training_session*training_steps, " out of ", training_sessions*training_steps,"...")
    classifier.train(input_fn=train_input_fn, steps=25)
    print("     Finished!")
    
    # Evaluate the model and print results to a file
    print("\nTesting model at step ", (training_session+1)*training_steps, " out of ", training_sessions*training_steps,"...")
    eval_result = classifier.evaluate(input_fn=test_input_fn, steps=None)
    print("     Finished!")
    
    file.write("Global Step:    " + str(eval_result["global_step"]) + "\n")
    file.write("    Average Loss:    " + str(eval_result["average_loss"]) + "\n")
    file.write("    Loss:            " + str(eval_result["loss"]) + "\n")
    file.write("    Label Mean:      " + str(eval_result["label/mean"]) + "\n")
    file.write("    Prediction Mean: " + str(eval_result["prediction/mean"]) + "\n\n")

# Close the file
file.close()