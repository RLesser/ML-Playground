from basicNN import train_neural_network
from sentiment_data_builder import create_feature_sets_and_labels
import numpy as np
# import pickle

# with open('sentiment_set.pickle','r') as f:
# 	pickle_data = pickle.load(f)

files = ['pos_sentiment.txt', 'neg_sentiment.txt']
labels = [[1,0], [0,1]]
train_x, train_y, test_x, test_y = create_feature_sets_and_labels(files, labels, max_grouping = 4, test_size = 0.1)

def average_accuracy(epoch_count, batch_size, hl_vec, n = 5):
	result_list = []
	for i in range(n):
		acc = train_neural_network([train_x, train_y], [test_x, test_y], 
			  epoch_count = epoch_count, batch_size = batch_size, hl_vec = hl_vec)
		result_list.append(acc)
	return np.mean(np.array(result_list)), np.median(np.array(result_list))

print average_accuracy(40, 50, [])

# epoch_tests = [10,20,40,80]
# hl_vec_tests = [[], [10], [20], [40], [80], [160], [320], [640], [1280]]
# mean_results = []
# median_results = []
# for e in epoch_tests:
# 	mean_list = []
# 	median_list = []
# 	for hl in hl_vec_tests:
# 		mean, median = average_accuracy(e, 50, hl)
# 		mean_list.append(mean)
# 		median_list.append(median)
# 	mean_results.append(mean_list)
# 	median_results.append(median_list)

# print mean_results
# print median_results

# mean_results
'''
0.5943715, 0.56998122, 0.57617259, 0.60938084, 0.61463416, 0.60750473, 0.59924948, 0.59962475, 0.60731709
0.64240152, 0.62457788, 0.62570357, 0.62570363, 0.61876172, 0.61181986, 0.59906185, 0.60712945, 0.60881793
0.67767352, 0.65534711, 0.6590994, 0.64784241, 0.61894929, 0.60487807, 0.60994369, 0.59643531, 0.61050653
0.67729837, 0.6521576, 0.64165103, 0.62288928, 0.62063789, 0.61557227, 0.61500943, 0.6060037, 0.6135084
'''
# median_results
'''
0.59287053, 0.57786119, 0.57410884, 0.61257035, 0.61069417, 0.60787994, 0.59849906, 0.597561, 0.59943718
0.64634144, 0.62945592, 0.6238274, 0.62945592, 0.6210131, 0.61444652, 0.59662288, 0.61163229, 0.60506564
0.67542213, 0.64821762, 0.66135085, 0.65384614, 0.61257035, 0.60694182, 0.61257035, 0.59474671, 0.61538464
0.67729831, 0.65196997, 0.64165103, 0.62288928, 0.62476546, 0.61538464, 0.62007505, 0.60694182, 0.61444652
'''

# time
# 7957s



 