from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import CNN_Regressor_model as cm

tf.logging.set_verbosity(tf.logging.INFO)

# Training and test sets
X_TRAINING = "../../../datasets/npy/train/x/train_x.npy"
Y_TRAINING = "../../../datasets/npy/train/y/regression/train_y_links_per_species_normalized.npy"
X_TEST = "../../../datasets/npy/test/x/test_x.npy"
Y_TEST= "../../../datasets/npy/test/y/regression/test_y_links_per_species_normalized.npy"

# Declare and open results file
RESULTS_FILE = "/Users/parkerkingfournier/Documents/Development/workspace/projects/Ecological-Inference/python/models/Regressors/logs/cnn_links_per_species_normalized/results.txt"
file = open(RESULTS_FILE, "w+")

# Load files
print("\nLoading files...")
train_x = np.load(X_TRAINING)
train_y = np.load(Y_TRAINING)
test_x  = np.load(X_TEST)
test_y  = np.load(Y_TEST)
print("     Finished!")

# Create the Estimator
classifier = tf.estimator.Estimator(model_fn=cm.cnn_model_fn, 
                                    model_dir="/Users/parkerkingfournier/Documents/Development/workspace/projects/Ecological-Inference/python/models/Regressors/logs/cnn_links_per_species_normalized")

# Set up logging for predictions
tensors_to_log = {"probabilities": "prediction_tensor/BiasAdd"}
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

# Training related variables
training_steps      = 50
training_sessions   = 40

for training_session in xrange(0,training_sessions):
    # Train the Model
    print("\nTraining model at step ", training_session*training_steps, " out of ", training_sessions*training_steps,"...")
    classifier.train( 
        input_fn=train_input_fn, 
        steps=50,
        hooks=[logging_hook])
    print("     Finished!\n")

    # Evaluate the model and print results to a file
    print("\nTesting model at step ", (training_session+1)*training_steps, " out of ", training_sessions*training_steps,"...")
    eval_result = classifier.evaluate(input_fn=eval_input_fn)
    print("     Finished!")

    file.write("Global Step:    " + str(eval_result["global_step"]) + "\n")
    file.write("    Loss:           " + str(eval_result["loss"]) + "\n")
    file.write("    Accuracy:       " + str(eval_result["accuracy"]) + "\n")

# Close the file
file.close()