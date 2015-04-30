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
Usage: speech_noise_ir_audio_mixing_script.py <speaker_speech_dir> <reference_file> <noise_file_dir> <ir_noise_file_dir> <output_root_directory>

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
Usage: input_and_target_dataset_generator.py <annotations_dir> <noise_mix_speech_dir> <mfcc_vector_output_file> <class_label_vector_output_file>

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
### Accuracy Scores

Support Vector Machines (10 Fold Cross Validation):  0.93413