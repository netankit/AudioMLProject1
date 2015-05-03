import scipy.io.wavfile as wav
import numpy as np
import math
from features import mfcc
import sys
import os
import cPickle
import scipy.sparse
from scipy import signal

if len(sys.argv)!=5:
    print '\nUsage: FrameLevelAnnotation_weighted_39D.py <annotations_dir> <noise_mix_speech_dir> <mfcc_vector_output_file> <class_label_vector_output_file>'
    sys.exit()

annotation_dir = sys.argv[1]
audio_files_dir= sys.argv[2]
mfcc_vector_output_file=sys.argv[3]
class_label_vector_output_file=sys.argv[4]

def getframelevelanno(noise_mix_speech_ano_file):
	#MFCC Parameters
	samplerate=16000
	winlen=0.025
	winstep=0.01
	winweight = signal.hamming(winlen*samplerate)
	winweight = winweight/sum(winweight)
	annotation = [line.strip() for line in open(noise_mix_speech_ano_file, 'r')]
	
	window = []
	frames_ano = []
	
	for bv in annotation:
		window.append(float(bv))
		if len(window)==winlen*samplerate:
			frames_ano.append(sum(np.multiply(window,winweight)))	
			window=window[int(winstep*samplerate):] #moving window
	frames_ano.append(sum(np.multiply(window,winweight[:len(window)]))) #the remaining samples for the last one frame
	return frames_ano



def getMfccVector(noise_mix_speech_file):
	(rate, signal) = wav.read(noise_mix_speech_file)
	mfcc_vec = mfcc(signal,rate,numcep=26)
	return mfcc_vec

def getDataset(annotation_dir, audio_files_dir):
	speech_vector_final = np.zeros((1,26))
	speech_vector_final = np.delete(speech_vector_final, (0), axis=0)
	print "Speech Vector Final: "+ str(speech_vector_final.shape)

	class_label_vector_final = []
	directoryCount = 1

	for root, dirs, files in os.walk(annotation_dir):
		path = root.split('/')
		for file in files:
			if (file.lower().endswith('.wav')):
				speechFilePath = os.path.join(root,str(file))
				tmp = os.path.dirname(speechFilePath)
				root_dir_name = os.path.basename(tmp)

				annotationfilename = str(os.path.splitext(file)[0])+'.ano'
				annotation_file_fullpath = os.path.join(annotation_dir,root_dir_name,annotationfilename)
				audio_file_fullpath = os.path.join(audio_files_dir,root_dir_name,file)

				print "Annotation File: "+ annotation_file_fullpath
				print "Audio file:"+ audio_file_fullpath

				mfcc_vector =  getMfccVector(audio_file_fullpath)
				class_label_vector = getframelevelanno(annotation_file_fullpath)
				(x,y) = mfcc_vector.shape
				z = len(class_label_vector)
				
				if (x==z):
					print "MFCC: " + str(x)
					print "LABEL: " + str(len(class_label_vector))
					print "speech_v: " + str(speech_vector_final.shape)
					print "mfcc_v: " + str(mfcc_vector.shape)
					speech_vector_final = np.vstack((speech_vector_final,mfcc_vector))
					class_label_vector_final.extend(class_label_vector)
		print "Directory Count: "+str(directoryCount)
		directoryCount = directoryCount+1

	return (speech_vector_final,class_label_vector_final)
				

#Main Routine
print "Start of the Program ....."
(speech_vector_final,class_label_vector_final) = getDataset(annotation_dir, audio_files_dir)



# Generate the various output files
#MFCC Speech
mfcc_vector_file = open(mfcc_vector_output_file, 'w')
temp1 = scipy.sparse.coo_matrix(speech_vector_final)
cPickle.dump(temp1,mfcc_vector_file,-1)
mfcc_vector_file.close()

#Class Labels
class_label_vector_file = open(class_label_vector_output_file, 'w')
class_label_vector_final_array = np.array(class_label_vector_final).reshape(len(class_label_vector_final),1)
temp2 = scipy.sparse.coo_matrix(class_label_vector_final_array)
cPickle.dump(temp2,class_label_vector_file,-1)
class_label_vector_file.close()

print "Final Shapes:"
print "Speech Vector:"+str(speech_vector_final.shape)
print "Class Labels:"+str(class_label_vector_final_array.shape)

print "Program completed Successfully"


