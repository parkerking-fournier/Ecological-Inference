import tensorflow as tf
import numpy as np

def cnn_model_fn(features, labels, mode):

	filter1 = tf.random_normal([40, 1, 20],   mean=0.0, stddev=0.05, dtype=tf.float64)
	filter2 = tf.random_normal([20, 20, 40],  mean=0.0, stddev=0.05, dtype=tf.float64)
	filter3 = tf.random_normal([10, 40, 80],  mean=0.0, stddev=0.05, dtype=tf.float64)
	filter4 = tf.random_normal([10, 80, 160], mean=0.0, stddev=0.05, dtype=tf.float64)
	filter5 = tf.random_normal([5, 160, 320], mean=0.0, stddev=0.05, dtype=tf.float64)
  	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, 149769, 1])

	# Convolutional Layer #1 and Pooling Layer #1		
	conv1 = tf.nn.conv1d(input_layer, filter1, stride=1, padding='SAME')
	pool1 = tf.layers.max_pooling1d(conv1, 43, 43, padding='SAME')
	
	# Convolutional Layer #2 and Pooling Layer #2			
	conv2 = tf.nn.conv1d(pool1, filter2, stride=1, padding='SAME')
	pool2 = tf.layers.max_pooling1d(conv2, 3, 3, padding='SAME')
	
	# Convolutional Layer #3 and Pooling Layer #3			
	conv3 = tf.nn.conv1d(pool2, filter3, stride=1, padding='SAME')
	pool3 = tf.layers.max_pooling1d(conv3, 3, 3, padding='SAME')
	
	# Convolutional Layer #4 and Pooling Layer #4			
	conv4 = tf.nn.conv1d(pool3, filter4, stride=1, padding='SAME')
	pool4 = tf.layers.max_pooling1d(conv4, 3, 3, padding='SAME')
	
	# Convolutional Layer #5 and Pooling Layer #5			
	conv5 = tf.nn.conv1d(pool4, filter5, stride=1, padding='SAME')
	pool5 = tf.layers.max_pooling1d(conv5, 3, 3, padding='SAME')
	
	# Dense Layer #1 and Droput Layer #1
	pool5_flat = tf.reshape(pool5, [-1, 43*320])
	dense1 = tf.layers.dense(inputs=pool5_flat, units=1024, activation=tf.nn.relu)
	dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
	
	# Logits Layer
	logits = tf.layers.dense(inputs=dropout1, units=3)

	predictions = {	"classes": tf.argmax(input=logits, axis=1),
					"probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)