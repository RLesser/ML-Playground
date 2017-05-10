import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

def reduce_string_list(string_list):
	lemmatizer = WordNetLemmatizer()
	#print len(string_list)
	string_list = [i for i in string_list if i[0].isalnum()]
	string_list = [lemmatizer.lemmatize(i) for i in string_list]
	#print len(string_list)
	#print string_list[:100]
	return string_list

def string_to_ngrams(line, max_grouping):
	token_list = word_tokenize(line.decode("utf8"))
	true_words = reduce_string_list(token_list)
	multi_words = []
	for group_size in range(1,max_grouping):
		multi_words += [" ".join(true_words[idx:idx+group_size+1]) for idx in range(len(true_words)-group_size)]
	true_words += multi_words
	#print true_words
	#exit()
	return true_words

def create_lexicon(file_list, max_grouping):
	lexicon = []
	for file in file_list:
		with open(file, 'r') as f:
			contents = f.readlines()
			for line in contents:
				ngram_list = string_to_ngrams(line, max_grouping)
				lexicon += list(ngram_list)
	word_counts = Counter(lexicon)
	high_cut = 1000000000
	low_cut = 0
	lexicon = [word for word in word_counts if word_counts[word] > 20] 
	# lower low_cut for more accuracy?
	print "lexicon created, size:", len(lexicon)
	return lexicon

def sample_handling(file, lexicon, classification, max_grouping):
	feature_set = []
	print "creating feature set for", file, "..."
	with open(file,'r') as f:
		contents = f.readlines()
		for line in contents:
			ngram_list = string_to_ngrams(line, max_grouping)
			features = np.zeros(len(lexicon))
			for word in ngram_list:
				if word in lexicon:
					index_value = lexicon.index(word)
					features[index_value] += 1
			features = list(features)
			feature_set.append([features, classification])
	print "feature set created, size:", len(feature_set)
	return feature_set

def create_feature_sets_and_labels(data_files,labels,
								   max_grouping = 1, test_size = 0.1):
	lexicon = create_lexicon(data_files, max_grouping)
	features = []
	for idx in range(len(data_files)):
		features += sample_handling(data_files[idx], lexicon, 
									labels[idx], max_grouping)

	random.shuffle(features) # needed?

	features = np.array(features)

	# number of items reserved for tests
	testing_size = int(test_size * len(features))

	train_x = list(features[:, 0][:-testing_size])
	train_y = list(features[:, 1][:-testing_size])
	print "training data created, size:", len(train_x)

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])
	print "testing data created, size:", len(test_x)

	return train_x, train_y, test_x, test_y
	

if __name__ == '__main__':
	file_list = ['pos_sentiment.txt', 'neg_sentiment.txt']
	label_list = [[1,0], [0,1]]
	train_x, train_y, test_x, test_y = \
		create_feature_sets_and_labels(file_list, label_list, max_grouping = 4)

	# with open('sentiment_set.pickle', 'wb') as f:
	# 	pickle.dump([train_x, train_y, test_x, test_y], f)

