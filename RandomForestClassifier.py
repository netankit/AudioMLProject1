from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import time
import sys
import cPickle as pickle
import scipy.sparse
from itertools import groupby
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#We have chosen l2 normalization to normalize the mfcc speech vector over the entire set of frames.
#Training Data -- Speech Vector File 
with open('class_label_weighted_vector_39D.dat', 'rb') as infile1:
   InputData = pickle.load(infile1)
InputDataSpeech = preprocessing.normalize(InputData,norm='l2')
infile1.close()

# Target Values -- Class Label Files.
with open('class_label_weighted_vector.dat', 'rb') as infile2:
   TargetData = pickle.load(infile2)
TargetClassLabelTemp = preprocessing.normalize(TargetData,norm='l2')
infile2.close()

TargetClassLabel = np.array(scipy.sparse.coo_matrix((TargetClassLabelTemp),dtype=np.int16).toarray()).tolist()
TargetClassLabel1 = map(str, TargetClassLabel)
TargetClassLabel = results = [int(i.strip('[').strip(']')) for i in TargetClassLabel1]

n_samples = len(TargetClassLabel)
InputDataSpeechTemp = scipy.sparse.coo_matrix((InputDataSpeech),dtype=np.float64).toarray()

# merging the label and mfcc feature
data_zipped = zip(TargetClassLabel, InputDataSpeechTemp)

# permutation of whole dataset
#data_zipped = np.random.permutation(data_zipped)

# seperating the ones and zeros
ones_index = [i for i, j in enumerate(TargetClassLabel) if j == 1]
zeros_index = [i for i, j in enumerate(TargetClassLabel) if j == 0]
ones = [data_zipped[i] for i in ones_index]
zeros = [data_zipped[i] for i in zeros_index]
num_zero = len(zeros)
original_ratio = float(len(ones))/num_zero
# number of zeros:  145082
# number of ones:  2057376

one_zero_ratio = 4
train_test_ratio = 10

# reconstructe test set
print 'The ratio of ones to zeros on test: ', one_zero_ratio
test_zero = zeros[:int(num_zero/train_test_ratio)]
print 'test of zeros:	', len(test_zero)
test_one = ones[:int(len(test_zero)*original_ratio)]
print 'test of ones:	', len(test_one)
print ''
test = test_zero + test_one
np.random.shuffle(test)
print 'test:	', len(test)
test_label = []
test_feature = []
for i in test:
	test_label.append(i[0])
for i in test:
	test_feature.append(i[1])

# reconstructe training set
# the ratio of ones to zeros on training is 1
train_zero = zeros[len(test_zero)+1:]
print 'train_zero:	', len(train_zero)
train_one = ones[len(test_one)+1:int(len(train_zero)*one_zero_ratio)]
print 'train_one	', len(train_one)
train = train_zero + train_one
np.random.shuffle(train)
print 'train:	', len(train)
train_label = []
train_feature = []
for i in train:
	train_label.append(i[0])
#print 'train_label', len(train_label)
for i in train:
	train_feature.append(i[1])
#print 'train_feature', len(train_feature)

clf_model = RandomForestClassifier(n_estimators=100).fit(train_feature, train_label)
label_predicted = clf_model.predict(test_feature)

target_names = ['0', '1' ]
print("Classification report for classifier %s:\n%s\n"
      % (clf_model, metrics.classification_report(test_label, label_predicted, target_names=target_names)))
cm = metrics.confusion_matrix(test_label, label_predicted)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Confusion matrix, without normalization')
print(cm) 
print('Normalized confusion matrix')
print(cm_normalized)
print 'Task is Finished!'
