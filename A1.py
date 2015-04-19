import scipy.io.wavfile as wav
import audiotools
import numpy
import math
import pydub
import random
from features import mfcc

# ReplayGain calculation of reference
ref = audiotools.open("ref_pink.wav")
ref_replay_gain = audiotools.calculate_replay_gain([ref])
ref_track_gain = list(list(ref_replay_gain)[0])[1]
print ref_track_gain

# ReplayGain calculation of example speech file
speech = audiotools.open('SA1_631.wav')
speech_replay_gain = audiotools.calculate_replay_gain([speech])
speech_track_gain = list(list(speech_replay_gain)[0])[1]
print speech_track_gain

# Normalization of example speech file
(rate_speech, data_speech) = wav.read('SA1_631.wav')
gain = ref_track_gain-speech_track_gain
data_normalized = numpy.asarray(data_speech*math.pow(10, (-(gain)/20)), dtype=numpy.int16)
wav.write('speech_normalized.wav', rate_speech, data_normalized)

# Loudness test of normalized example speech
test = audiotools.open('speech_normalized.wav')
test_replay_gain = audiotools.calculate_replay_gain([test])
test_track_gain = list(list(test_replay_gain)[0])[1]
print test_track_gain

# Randomly choosing one noise file from the pool
# here I just fix one waiting for implementation later

# Using pydub API to calculate the length of normalized speech file and the noise file
speech_normalized = pydub.AudioSegment.from_wav("speech_normalized.wav")
# !there is a bug of this function: can NOT open some noise files
noise = pydub.AudioSegment.from_wav("super_market_mall2.wav")
speech_normalized_length = speech_normalized.duration_seconds
noise_length = noise.duration_seconds

# Selecting a randow start point of the noise file to get a segment of the required length
start = random.randrange(0,int(noise_length-speech_normalized_length)*1000)
# pydub does things in milliseconds
noise_segmented = noise[start:int(start+speech_normalized_length*1000)]
noise_segmented.export("noise_segmented.wav",format="wav")

# Linear fading of sharply segmented noised segment
# 1 sec fade in, 1 sec fade out
noise_faded = noise_segmented.fade_in(1000).fade_out(1000)
noise_faded.export("noise_faded.wav",format="wav")

# how long is good? 1 sec?

# Picking a random signal to noise ratio (SNR)
SNR_ratio = random.randint(-2, 20)
print "SNR_ratio: " + str(SNR_ratio)

# loudness in dBFS (Decibels relative to full scale)
# (all peak measurements will be negative numbers)
speech_dB = speech_normalized.dBFS
noise_dB = noise_segmented.dBFS
print "loudness of speech: " + str(speech_dB)
print "loudness of noise: " + str(noise_dB)

# Change the amplitude (generally, loudness) of the speech by SNR ratio from noise. 
# Gain is specified in dB. 
gain = SNR_ratio-(speech_dB-noise_dB)
print "gain: " + str(gain)
speech_SNRed = speech_normalized.apply_gain(gain)
print "loudness of adjusted speech: " + str(speech_SNRed.dBFS)
# check SNR
print "check SNR: " + str(speech_SNRed.dBFS - noise_dB)

# mix the two tracks by adding the respective samples
# (If the overlaid AudioSegment is longer than this one, the result will be truncated)
noisy_speech = speech_SNRed.overlay(noise_segmented)
noisy_speech.export("noisy_speech.wav",format="wav")
# Since the sample values have increased through the summation, it is possible that they exceed the maximum imposed by the data type. How this API deals with this problem?


# draw an impulse response from the pool
# ...waiting to implement

# peak-normalize it to 0dB (=1) by dividing the IR vector through its maximum value.
(rate_IR, data_IR) = wav.read("htc_desk.wav")
# data_IR.dtype is int16, change it into float64
data_IR = data_IR.astype(numpy.float64) 
data_IR = data_IR / data_IR.max()

# convolve speech with the normalized IR
(rate_noisy_speech, data_noisy_speech) = wav.read("noisy_speech.wav")
speech_convolved = numpy.convolve(data_IR, data_noisy_speech)

# cut the convolved track to its original length if prolonged and store the resulting track
wav.write('speech_convolved.wav', rate_noisy_speech, speech_convolved[:data_noisy_speech.size])

# MFCC Feature extraction
# Do the default parameters (frame size etc.) work for you?
(rate,sig) = wav.read("speech_convolved.wav")
mfcc_feat = mfcc(sig,rate)
print mfcc_feat.shape
print mfcc_feat








