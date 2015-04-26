Audio Based Machine Learning with Python Assignment I
-----------------------------------------------------

1. A1.py - Generation of Convoluted Speech File for a single input speech file.

2. beta_script.py - Generates the Convoluted speech files in a given "output" directory, with customized command line parameters.

Usage: beta_script.py <pathtodir> <referenceFile> <noiseFile> <output_root_directory>

Example Run (On Test Server, Sample is available at /mnt/alderaan/mlteam3/A1):
python beta_script.py ../data/sample/ ref_pink.wav super_market_mall2.wav ~/output/

Sample Output: 
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

