from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import CNN_LSTM_Regressor_model as cm_lm

tf.logging.set_verbosity(tf.logging.INFO)

# Training and test sets
X_TRAINING = "../../../datasets/npy/train/x/train_x.npy"
Y_TRAINING = "../../../datasets/npy/train/y/regression/train_y_links_per_species.npy"
X_TEST = "../../../datasets/npy/test/x/test_x.npy"
Y_TEST= "../../../datasets/npy/test/y/regression/test_y_links_per_species.npy"

# Load files
print("\nLoading files...")
test_x  = np.load(X_TEST)
test_y  = np.load(Y_TEST)
test_y  = test_y.astype(int)
print("     Finished!")

# Create the Estimator
classifier = tf.estimator.Estimator(model_fn=cm_lm.cnn_lstm_model_fn, 
                                    model_dir="/Users/parkerkingfournier/Documents/Development/workspace/projects/Ecological-Inference/python/models/Regressors/logs/cnn_lstm_links_per_species")

# Set up logging for predictions
tensors_to_log = {"probabilities": "prediction_tensor/BiasAdd"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# Define training and eval inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn( 
    x={"x": train_x}, 
    y=train_y, 
    batch_size=50, 
    num_epochs=None, 
    shuffle=True)

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