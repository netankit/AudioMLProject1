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


if len(sys.argv)!=3:
    print '\nUsage: python classification_report.py <speech_vector_file> <class_label_file>'
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

#Choosing SVM as our machine learning model.
#clf_model = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', multi_class='ovr', penalty='l2',random_state=None, tol=0.0001, verbose=0)
clf_model = svm.LinearSVC()

print 'Starting Cross Validation'

n_samples = len(TargetClassLabel)
InputDataSpeechTemp = scipy.sparse.coo_matrix((InputDataSpeech),dtype=np.float64).toarray()
first_half_input_vector = InputDataSpeechTemp[:n_samples/2,:13]
second_half_input_vector = InputDataSpeechTemp[n_samples/2:,:13]
#print "Predicted: "+str(second_half_input_vector.shape)

clf_model.fit(first_half_input_vector, TargetClassLabel[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = TargetClassLabel[n_samples / 2:]
predicted = clf_model.predict(second_half_input_vector)
#predicted = predicted1.tolist()
#predicted[0] = 0

target_names = ['No Speech Detected', 'Speech Detected' ]

#Generate Classification Report
print("Classification report for classifier %s:\n%s\n"
      % (clf_model, metrics.classification_report(expected, predicted,target_names=target_names)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

#Recording the end time.
end = time.time()
print "Total execution time in minutes :: >>"
print (end - start)/60
print 'Task is Finished!'

# print "Expected: "+str([len(list(group)) for key, group in groupby(expected)])
# print "Predicted: "+str([len(list(group)) for key, group in groupby(predicted)])

#print "Expected: "+str(expected)
#print "Predicted: "+str(predicted)
