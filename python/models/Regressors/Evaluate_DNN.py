from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Test sets
X_TEST = "../../../datasets/npy/test/x/test_x.npy"
Y_TEST= "../../../datasets/npy/test/y/regression/test_y_characteristic_length.npy"

# Load files
print("Loading files...")
test_x  = np.load(X_TEST)
test_y  = np.load(Y_TEST)
print("     Finished!")

# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[149769])]

# Build 5 layer DNN with 50, 50, 50, 50, 50 units respectively.
classifier = tf.estimator.DNNRegressor( feature_columns=feature_columns,
                                        hidden_units=[50, 50, 50, 50, 50],
                                        model_dir="/Users/parkerkingfournier/Documents/Development/workspace/projects/Ecological-Inference/python/scripts/models/Regressors/logs/dnn_characteristic_length",
                                        dropout=0.5)

# Define the eval inputs
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_x},
    y=test_y,
    num_epochs=1,
    shuffle=False)

# Initialize performance evaluation metrics
limit           = 1
label_mean      = 0
average_loss    = 0
prediction_mean = 0
loss            = 0

#Evaluate the model and print results
print("\nTesting model...\n")
for i in xrange(0,limit):
    print("\n Performing test ", (i+1), " of ", limit, "\n")
    eval_result     = classifier.evaluate(input_fn=eval_input_fn)
    label_mean      += eval_result["label/mean"]
    average_loss    += eval_result["average_loss"]
    prediction_mean += eval_result["prediction/mean"]
    loss            += eval_result["loss"]

# Find the average of the evaluation metrics over all tests
average_label_mean      = label_mean/limit
average_average_loss    = average_loss/limit
average_prediction_mean = prediction_mean/limit
average_loss            = loss/limit

print("\nFinished!\n")

# Print results
print("\naverage_label_mean:      ", average_label_mean)
print("average_average_loss:    ", average_average_loss)
print("average_prediction_mean: ", average_prediction_mean)
print("average_loss:            ", average_loss, "\n")















