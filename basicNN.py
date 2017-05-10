import tensorflow as tf

def make_hl_obj_list(input_size, hl_size_vec, n_out_size):
	make_hl_obj = lambda i_nodes, o_nodes: \
		{'weights': tf.Variable(tf.random_normal([i_nodes, o_nodes])),         
		'biases': tf.Variable(tf.random_normal([o_nodes]))}
	totalVec = [input_size] + hl_size_vec + [n_out_size]
	return [make_hl_obj(totalVec[x], totalVec[x+1])
			for x in range(len(totalVec)-1)]

def neural_network_model(data, in_size, out_size, hl_vec):
	hl_obj_list = make_hl_obj_list(in_size, hl_vec, out_size)

	layer_data_list = [data]
	for hl_idx in range(len(hl_obj_list)):
		cur_input_data = layer_data_list[hl_idx]
		cur_layer = hl_obj_list[hl_idx]
		# first, mul the prev output data (or starting data) with layer weights
		layer_data = tf.matmul(cur_input_data, cur_layer['weights'])
		# next, add that product with the biases for the layer
		layer_data = tf.add(layer_data, cur_layer['biases'])
		
		# as long as not output layer, do relu on layer_data
		if hl_idx != len(hl_obj_list)-1:
			layer_data = tf.nn.relu(layer_data)

		# store layer data for the next layer's analysis
		layer_data_list.append(layer_data)

	# return the output layer data
	return layer_data_list[-1]

def train_neural_network(train, test, epoch_count = 10, batch_size = 100, hl_vec = [500,500,500]):
	x = tf.placeholder('float', [None, len(train[0][0])])
	y = tf.placeholder('float')
	in_size = len(train[0][0])
	out_size = len(train[1][0])
	prediction = neural_network_model(x, in_size, out_size, hl_vec)
	softmax = tf.nn.softmax_cross_entropy_with_logits(prediction,y)
	cost = tf.reduce_mean(softmax)
	# default learning rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		for epoch in range(epoch_count):
			epoch_loss = 0
			for idx in range(int(len(train[0]) / batch_size)):
				batch_x = train[0][batch_size*idx : batch_size*(idx+1)]
				batch_y = train[1][batch_size*idx : batch_size*(idx+1)]

				idx, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_loss += c
			print 'Epoch', epoch+1, 'completed out of', epoch_count,
			print 'loss:', epoch_loss
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy:', accuracy.eval({x:test[0], 
										  y:test[1]})
		return accuracy.eval({x:test[0], y:test[1]})

if __name__ == '__main__':
	print """
	Basic Neural Network implimented with tensorflow

	train_neural_network() takes the following arguments
	train - this is the training data and labels, in the form [data list, label list]
	test - this is the testing data and labels, in the form [data list, label list]
	batch_size (default = 100) - this is the size of the batches to be created
	epoch_count (default = 10) - this is the number of epochs (or cycles) to run
	hl_vec (default = [500,500,500]) - this is the neural network shape, with each 
	entry being a layer and the number in each entry being the nodes in that layer
	"""




