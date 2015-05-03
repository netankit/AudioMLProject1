from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import time
import sys
import cPickle as pickle
import scipy.sparse
from itertools import groupby


if len(sys.argv)!=3:
    print '\nUsage: python estimate_svm_parameters.py <speech_vector_file> <class_label_file>'
    sys.exit()

speech_vector_file = sys.argv[1]
class_label_file = sys.argv[2]
#cross_validation_folds_number = sys.argv[3]

#We have chosen l2 normalization to normalize the mfcc speech vector over the entire set of frames.
#Training Data -- Speech Vector File 
with open(speech_vector_file, 'rb') as infile1:
   InputData = pickle.load(infile1)
InputDataSpeech = preprocessing.normalize(InputData,norm='l2')
infile1.close()

# Target Values -- Class Label Files.
with open(class_label_file, 'rb') as infile2:
   TargetData = pickle.load(infile2)
TargetClassLabelTemp = preprocessing.normalize(TargetData,norm='l2')
infile2.close()

#print InputDataSpeech.shape
#print TargetClassLabelTemp.shape

TargetClassLabel = np.array(scipy.sparse.coo_matrix((TargetClassLabelTemp),dtype=np.int16).toarray()).tolist()
TargetClassLabel1 = map(str, TargetClassLabel)
#TargetClassLabel = map(int, TargetClassLabel)
TargetClassLabel = results = [int(i.strip('[').strip(']')) for i in TargetClassLabel1]

#Recording the start time.
start = time.time()

# Loading the Digits dataset
#digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
#X = digits.images.reshape((n_samples, -1))
#y = digits.target

# Split the dataset in two equal parts
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.5, random_state=0)

n_samples = len(TargetClassLabel)
InputDataSpeechTemp = scipy.sparse.coo_matrix((InputDataSpeech),dtype=np.float64).toarray()
X_train = InputDataSpeechTemp[:n_samples/2,:13]
X_test = InputDataSpeechTemp[n_samples/2:,:13]
y_train = TargetClassLabel[:n_samples / 2]
y_test =  TargetClassLabel[n_samples / 2:]

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {{'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4], 
                     'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)

    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

#Recording the end time.
end = time.time()
print "Total execution time in minutes :: >>"
print (end - start)/60
print 'Task is Finished!'
