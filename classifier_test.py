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


#We have chosen l2 normalization to normalize the mfcc speech vector over the entire set of frames.
#Training Data -- Speech Vector File 
with open('mfcc_full_vector.dat', 'rb') as infile1:
   InputData = pickle.load(infile1)
InputDataSpeech = preprocessing.normalize(InputData,norm='l2')
infile1.close()

# Target Values -- Class Label Files.
with open('class_label_full_vector.dat', 'rb') as infile2:
   TargetData = pickle.load(infile2)
TargetClassLabelTemp = preprocessing.normalize(TargetData,norm='l2')
infile2.close()

#print InputDataSpeech.shape
#print TargetClassLabelTemp.shape

TargetClassLabel = np.array(scipy.sparse.coo_matrix((TargetClassLabelTemp),dtype=np.int16).toarray()).tolist()
TargetClassLabel1 = map(str, TargetClassLabel)
#TargetClassLabel = map(int, TargetClassLabel)
TargetClassLabel = results = [int(i.strip('[').strip(']')) for i in TargetClassLabel1]

n_samples = len(TargetClassLabel)
InputDataSpeechTemp = scipy.sparse.coo_matrix((InputDataSpeech),dtype=np.float64).toarray()

sub_feature = InputDataSpeechTemp[:n_samples/5,:13]
sub_label = TargetClassLabel[:n_samples / 5]
test_feature = InputDataSpeechTemp[99*n_samples/100:,:13]
test_label = TargetClassLabel[99*n_samples/100:]

# rearranging training set
ones_index= [i for i, j in enumerate(sub_label) if j == 1]
zeros_index = [i for i, j in enumerate(sub_label) if j == 0]
ones_index = np.random.permutation(ones_index)
zeros_index= np.random.permutation(zeros_index)
sub_zero = [sub_feature[i] for i in zeros_index]
sub_one = [sub_feature[i] for i in ones_index]
print str(len(sub_zero)) + ' 0' 
print str(len(sub_one)) + ' 1'
one_cutted = sub_one[:len(sub_zero)]
feature_balanced = np.concatenate((sub_zero, one_cutted), axis=0)
label_balanced = np.append(np.zeros(len(zeros_index)),np.ones(len(one_cutted)))

# rearranging test set FOR TEST
ones_index_test= [i for i, j in enumerate(test_label) if j == 1]
zeros_index_test = [i for i, j in enumerate(test_label) if j == 0]
ones_index = np.random.permutation(ones_index_test)
zeros_index= np.random.permutation(zeros_index_test)
test_zero = [test_feature[i] for i in zeros_index_test]
test_one = [test_feature[i] for i in ones_index_test]
print str(len(test_zero)) + ' 0' 
print str(len(test_one)) + ' 1'
one_cutted_test = test_one[:len(test_zero)]
feature_balanced_test = np.concatenate((test_zero, one_cutted_test), axis=0)
label_balanced_test = np.append(np.zeros(len(test_zero)),np.ones(len(one_cutted_test)))

clf_model = KNeighborsClassifier().fit(feature_balanced, label_balanced)
label_predicted = clf_model.predict(feature_balanced_test)

target_names = ['No Speech Detected', 'Speech Detected' ]
print("Classification report for classifier %s:\n%s\n"
      % (clf_model, metrics.classification_report(label_balanced_test, label_predicted,target_names=target_names)))
cm = metrics.confusion_matrix(label_balanced_test, label_predicted)
print label_balanced_test
print label_predicted
print len(label_balanced_test)
print len(label_predicted)
print sum(label_balanced_test)
print sum(label_predicted)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Confusion matrix, without normalization')
print(cm)
print('Normalized confusion matrix')
print(cm_normalized)
print 'Task is Finished!'
