import scipy.io.wavfile as wav
import numpy as np
import math
from features import mfcc

if len(sys.argv)!=2:
    print '\nUsage: FrameLevelAnnotation.py <annotations_dir> <noise_mix_speech_dir>'
    sys.exit()

annotation_dir = sys.argv[1]
audio_files_dir= sys.argv[2]

def getframelevelanno(noise_mix_speech_ano_file):
	#MFCC Parameters
	samplerate=16000
	winlen=0.025
	winstep=0.01
	
	annotation = [line.strip() for line in open(noise_mix_speech_ano_file, 'r')]
	
	window = []
	frames_ano = []
	
	for bv in annotation:
		window.append(float(bv))
		if len(window)==winlen*samplerate:
			frames_ano.append(math.ceil(sum(window)/len(window)))		
			window=window[int(winstep*samplerate):] #moving window

	frames_ano.append(math.ceil(float(sum(window))/len(window))) #the remaining samples for the last one frame
	frames_ano_array = np.transpose(np.array(sentencevector))
	return frames_ano_array



def getMfccVector(noise_mix_speech_file):
	(rate, signal) = wav.read(noise_mix_speech_file)
	mfcc = mfcc(signal,rate)
	return mfcc

def getDataset(annotation_dir, audio_files_dir):
	for root, dirs, files in os.walk(audio_files_dir):
		directoryCount = 1
		print "Directory Count: "+str(directoryCount)
		path = root.split('/')
	    
	    for file in files:
			if (file.lower().endswith('.wav')):
	        	annotation_file_fullpath = os.path.join(annotation_dir,file)
	        	audio_file_fullpath = os.path.join(audio_files_dir,file)
	        	print "Annotation File: "+ annotation_file_fullpath
	        	print "Audio file:"+ audio_file_fullpath
	            mfcc_vector =  getMfccVector(audio_file_fullpath)
				class_label_vector = getframelevelanno(annotation_file_fullpath)
				(x,y) = mfcc_vector.shape
				(z,w) = class_label_vector.shape
				if(x!=z):
					print "Failed!"
					sys.exit()





print "Start of the Program ....."
getDataset(annotation_dir, audio_files_dir)
print "Program completed Successfully"


