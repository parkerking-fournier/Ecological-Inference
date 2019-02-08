from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import CNN_model as cm

tf.logging.set_verbosity(tf.logging.INFO)

# Training and test sets
X_TRAINING = "../../../../datasets/npy/train_x.npy"
Y_TRAINING = "../../../../datasets/npy/train_y_num_species.npy"
X_TEST = "../../../../datasets/npy/test_x.npy"
Y_TEST = "../../../../datasets/npy/test_y_num_species.npy"

# Load files
print("\nLoading files...")
train_x = np.load(X_TRAINING)
train_y = np.load(Y_TRAINING)
train_y = train_y.astype(int)
test_x  = np.load(X_TEST)
test_y  = np.load(Y_TEST)
test_y = test_y.astype(int)
print("     Finished!")

# Create the Estimator
classifier = tf.estimator.Estimator(model_fn=cm.cnn_model_fn, 
                                    model_dir="/Users/parkerkingfournier/Documents/University/U4(2017-18)/Summer/ResearchProject/python/scripts/models/logs/classifiers/cnn_classifier")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# Define the training and eval inputs
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

# Train the Model
print("\nTraining model...\n")
classifier.train( 
    input_fn=train_input_fn, 
    steps=200, 
    hooks=[logging_hook])
print("     Finished!\n")

# Evaluate the model and print results
print("\nTesting model...\n")
eval_results = classifier.evaluate(input_fn=eval_input_fn)
print("     Finished!")

print("\nTest error:  ", eval_results, '\n')