## Audio Based Machine Learning in Python: Assignment 1

*Goals*
* Mix speech corpora with noise sounds; fold with impulse responses
* Extract MFCC features
* Voice Activity Detection using out-of-the-box classifiers (Random Forests, SVM, Neural Networks etc.) with cross validation
* Present your results

*Details and constraints*

In this first, introductory assignment you will create a dataset that simulates speech in every-day scenarios. You will train a classifier on this dataset for distinguishing voiced from non-voiced sections, a task called voice activity detection, VAD for short. This, of course, requires a ground truth in terms of VAD annotations.

### A1.py 
Generation of Convoluted Speech File for a single input speech file.

### beta_script.py
Generates the Convoluted speech files in a given "output" directory, with customized command line parameters.

```
Usage: beta_script.py <pathtodir> <referenceFile> <noiseFile> <output_root_directory>
```
Example Run (On Test Server, Sample is available at /mnt/alderaan/mlteam3/A1):
python beta_script.py ../data/sample/ ref_pink.wav super_market_mall2.wav ~/output/

Sample Output (Before dividing by 65536.0): 
```
Current File:  SA2_632.wav
6.0
9.9
6.0
SNR_ratio: 8
loudness of speech: -31.262838506
loudness of noise: -25.613592796
gain: 13.64924571
loudness of adjusted speech: -17.6134095363
check SNR: 8.0001832597
Final output file path: /mnt/alderaan/mlteam3/output/sample/convoluted/SA2_632.wav
MFCC Shape:
(262, 13)
[[ 21.36008246   4.68758863 -21.02879502 ..., -26.48962835   6.03351536
  -13.4272054 ]
 [ 20.31307126  -1.92123373  -4.03121067 ..., -32.55583596  -1.99548691
  -12.51953057]
 [ 20.33447627  -4.48978457  -1.50634631 ..., -37.70311405  -2.25513418
   -6.88467421]
 ...,
 [ 19.8452419    1.90521342  -7.74598993 ..., -25.54148603  10.53801712
  -15.36899988]
 [ 19.67552457  -2.96859201  -9.66682243 ..., -20.83320987   9.90071342
  -10.93574823]
 [ 19.18471866  -2.01026595  -9.93172322 ..., -27.09888399  -0.2349532
  -10.09651341]]
Operations Finished!
```

Sample Output (After dividing by 65536.0): 
```
mlteam3@pcschlichter4:~/A1$ python beta_script.py ../data/sample/ ref_pink.wav super_market_mall2.wav ~/output/
Current File:  SA2_632.wav
6.0
9.9
6.0
SNR_ratio: 3
loudness of speech: -31.262838506
loudness of noise: -24.2493999642
gain: 10.0134385417
loudness of adjusted speech: -21.2487508768
check SNR: 3.00064908746
Final output file path: /mnt/alderaan/mlteam3/output/sample/convoluted/SA2_632.wav
MFCC Shape:
(262, 13)
[[ 19.59508939   2.70186776  -9.23936962 ..., -34.90854563  -6.3798238
   -6.01107859]
 [ 19.55317435   1.11851098  -9.86874784 ..., -27.75712167   1.04552466
   -1.52660555]
 [ 19.57381244   3.3010208   -8.2860773  ..., -25.3561603    8.64244945
    0.60929827]
 ...,
 [ 19.22814775   4.87784517  -6.85796621 ..., -27.2970513   -6.59483469
  -13.7599967 ]
 [ 19.16051337   7.38684259  -7.06352436 ..., -19.57186845  -0.52468905
   -9.31084754]
 [ 18.98050344   7.67656769  -6.82664631 ..., -22.64922645  -7.03116873
  -13.19350393]]
Operations Finished!
```

### gamma_script.py

Generates the Convoluted speech files in a given "output" directory, with customized command line parameters.
```
Usage: gamma_script.py <speaker_speech_dir> <reference_file> <noise_file_dir> <ir_noise_file_dir> <output_root_directory>

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
Noise File Path: ../data/noise_sample/super_market_mall2.wav
IR-Noise File Path: /mnt/tatooine/data/impulse_responses/16kHz/wavs16b/s1_desk.wav
Final output file path: ../newout1/sample/SA2_632.wav
Operations Finished!

mlteam3@pcschlichter4:~/A1$ python gamma_script.py ../data/sample ref_pink.wav 
../data/noise_sample/ /mnt/tatooine/data/impulse_responses/16kHz/wavs16b ../newout1
Noise File Path: ../data/noise_sample/super_market_mall2.wav
IR-Noise File Path: /mnt/tatooine/data/impulse_responses/16kHz/wavs16b/s3_desk.wav
Final output file path: ../newout1/sample/SA2_632.wav
Operations Finished!

mlteam3@pcschlichter4:~/A1$ python gamma_script.py ../data/sample ref_pink.wav 
../data/noise_sample/ /mnt/tatooine/data/impulse_responses/16kHz/wavs16b ../newout1
Noise File Path: ../data/noise_sample/super_market_mall2_copy.wav
IR-Noise File Path: /mnt/tatooine/data/impulse_responses/16kHz/wavs16b/lg_leather_bag.wav
Final output file path: ../newout1/sample/SA2_632.wav
Operations Finished!
```

###svm_modelling.py

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
Final Accuracy Score: 0.93 (+/- 0.00)
Total execution time in minutes :: >>
7.22843818267
Task is Finished!
```
