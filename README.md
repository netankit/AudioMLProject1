## Audio Based Machine Learning in Python: Assignment 1

*Goals*
* Mix speech corpora with noise sounds; fold with impulse responses
* Extract MFCC features
* Voice Activity Detection using out-of-the-box classifiers (Random Forests, SVM, Neural Networks etc.) with cross validation
* Present your results

*Details and constraints*

In this first, introductory assignment you will create a dataset that simulates speech in every-day scenarios. You will train a classifier on this dataset for distinguishing voiced from non-voiced sections, a task called voice activity detection, VAD for short. This, of course, requires a ground truth in terms of VAD annotations.

### format_convertor.py

Converts the format of the noise files into int_16 type. We convert all the noise files into the int_16 format aand save them in a new folder which is used further in the generation of dataset.

```
Usage: python format_convertor.py
```

### speech_noise_ir_audio_mixing_script.py

Generates the Convoluted speech files in a given "output" directory, with customized command line parameters.
```
Usage: python speech_noise_ir_audio_mixing_script.py <speaker_speech_dir> <reference_file> <noise_file_dir> <ir_noise_file_dir> <output_root_directory>

Command line arguments:
<speaker_speech_dir>: Root directory where all the VAD speaker speech data is located
<reference_file>: Reference File for the Calculation of Replay gain, named as "ref_pink.wav"
<noise_file_dir>: Root directory where all the noise data is located
<ir_noise_file_dir> : Root directory where all the impulse response noise samples is located
<output_root_directory>: Root directory for storing all the output files in same directory hierarchy as the input speech directory
```

Sample Run and Output:
```
mlteam3@pcschlichter4:~/A1$ python gamma_script.py ../data/sample ref_pink.wav 
../data/noise_sample/ /mnt/tatooine/data/impulse_responses/16kHz/wavs16b ../newout1
Noise File Path: ../data/noise_sample/super_market_mall2_copy.wav
IR-Noise File Path: /mnt/tatooine/data/impulse_responses/16kHz/wavs16b/lg_leather_bag.wav
Final output file path: ../newout1/sample/SA2_632.wav
Operations Finished!
```

### input_and_target_dataset_generator.py

Generates the input and target label dataset from the given noise mixed speech files and corresponding annotation files. Stores them in scipy.sparse.coo_matrix representation and saves the resultant dataset vectors onto disk. 

```
Usage: python input_and_target_dataset_generator.py <annotations_dir> <noise_mix_speech_dir> <mfcc_vector_output_file> <class_label_vector_output_file>

```

Sample run:

```
mlteam3@pcschlichter4:~/A1$ python FrameLevelAnnotation.py ../data/vad_dirsample/ ../output_all/ mfcc_full_vector.dat class_label_full_vector.dat
```

### svm_modelling.py

This script, employs scikit learn's lib linear version of support vector machine and computes cross validation accuracy scores.

```
Usage: python svm_modelling.py <speech_vector_file> <class_label_file>
```
_CROSS Validation: 3_
```
mlteam3@pcschlichter4:~/A1$ python svm_modelling.py mfcc_full_vector.dat class_label_full_vector.dat
(2202458, 13)
(2202458, 1)
Starting Cross Validation
Final Accuracy Score: 0.93 (+/- 0.00)
Total execution time in minutes :: >>
1.27803570032
Task is Finished!
```

_CROSS Validation: 10_
```
mlteam3@pcschlichter4:~/A1$ python svm_modelling.py mfcc_full_vector.dat class_label_full_vector.dat
(2202458, 13)
(2202458, 1)
Starting Cross Validation
Final Accuracy Score: 0.93413 (+/- 0.00)
Total execution time in minutes :: >>
5.50397278468
Task is Finished!

```
### classification_report.py

Generates classification report for the given model. [STILL TO Investigate/ FEW BUGS]
```
Usage: python classification_report.py <speech_vector_file> <class_label_file>
```

Sample Run:

```
mlteam3@pcschlichter4:~/A1$ python classification_report.py mfcc_full_vector.dat class_label_full_vector.dat
(2202458, 13)
(2202458, 1)
Starting Cross Validation
/usr/local/lib/python2.7/dist-packages/sklearn/metrics/classification.py:958: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
Classification report for classifier LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0):
                    precision    recall  f1-score   support

No Speech Detected       0.00      0.00      0.00     60627
   Speech Detected       0.94      1.00      0.97   1040602

       avg / total       0.89      0.94      0.92   1101229


Confusion matrix:
[[      0   60627]
 [      0 1040602]]
Total execution time in minutes :: >>
0.405680398146
Task is Finished!

```

### Accuracy Scores

Support Vector Machines (10 Fold Cross Validation):  0.93413

### RandomForestClassifier.py

Seperate the speech frames and nonspeech frames;
Rearrange and shuffle the training set according to adjustable speech to non-speech frames ratio and using RandomForestClassifier.
```
Usage: python classification_report.py <speech_vector_file> <class_label_file>
```

Sample Run:

```
mlteam3@pcschlichter4:~/A1$ python RandomForestClassifier.py
test of zeros:  4836
test of ones:   68578
test:   73414
train_zero:     140245
train_one       140245
train:  280490
Classification report for classifier RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False):
             precision    recall  f1-score   support

          0       0.07      0.99      0.12      4836
          1       0.96      0.02      0.04     68578

avg / total       0.90      0.08      0.04     73414


Confusion matrix, without normalization
[[ 4779    57]
 [67331  1247]]
Normalized confusion matrix
[[ 0.9882134   0.0117866 ]
 [ 0.98181633  0.01818367]]
Task is Finished!


```
### estimate_svm_paramters.py and  estimate_liblinear_svm_parameters.py

This script(s) determines the best paramters for SVM and LinearSVC(LibLinear SVM implementation) using Grid search cross validation over five folds. 

```
Usage: python estimate_svm_parameters.py <speech_vector_file> <class_label_file>

Usage: estimate_liblinear_svm_parameters.py <speech_vector_file> <class_label_file>

```


The following parameters were experimented on, to find the most suitable configuration:

```
For SVM:
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                    {{'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4], 
                     'C': [1, 10, 100, 1000]}]
For LinearSVC:

tuned_parameters = [{'C': [1, 10, 100, 1000],'loss':['hinge' , 'squared_hinge'] }]


```

