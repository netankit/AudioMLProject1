from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
import numpy
import time
import sys
import cPickle as pickle


if len(sys.argv)!=4:
    print '\nUsage: python svm_modelling.py <speech_vector_file> <class_label_file> <cross_validation_folds_number>'
    sys.exit()

speech_vector_file = sys.argv[1]
class_label_file = sys.argv[2]
cross_validation_folds_number = sys.argv[3]

#We have chosen l2 normalization to normalize the mfcc speech vector over the entire set of frames.
#Training Data -- Speech Vector File 
with open(speech_vector_file, 'rb') as infile1:
   InputData = pickle.load(infile1)
InputDataSpeech = preprocessing.normalize(InputData,norm='l2')
infile1.close()

# Target Values -- Class Label Files.
with open(class_label_file, 'rb') as infile2:
   TargetData = pickle.load(infile2)
TargetClassLabel = preprocessing.normalize(TargetData,norm='l2')
infile2.close()

print InputDataSpeech.shape
print TargetClassLabel.shape


#Recording the start time.
start = time.time()

#Choosing SVM as our machine learning model.
clf_model = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',random_state=None, tol=0.0001, verbose=0)
print 'Starting Cross Validation'

# fit() is not required when cross validating.
#clf = clf_model.fit(InputDataSpeech,TargetClassLabel)

scores = cross_validation.cross_val_score(clf, InputDataSpeech, TargetClassLabel, cv=cross_validation_folds_number)
print "\nFinal Accuracy Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

#Recording the end time.
end = time.time()


print "Total execution time in minutes :: >>"
print (end - start)/60

print 'Task is Finished!'