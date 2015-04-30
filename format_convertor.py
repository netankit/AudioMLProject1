import os

NoiseFileDir = "/mnt/tatooine/data/noise/noise_equal_concat/train"
outputDir = "/mnt/alderaan/mlteam3/data/noise_int16"
for root, dirs, files in os.walk(NoiseFileDir):
    path = root.split('/')
    #print (len(path) - 1) *'---' , os.path.basename(root)       
    for file in files:
        if (file.lower().endswith('.wav')):
            #ir_file_list.append(file)     
            (rate_IR, data_IR) = wav.read(file)
			data_IR = data_IR.astype(numpy.int16)
			outputfilename = os.path.basename(file)
			outputfilepath = os.path.join(outputDir,outputfilename) 
			wav.write(outputfilepath, rate_IR, data_IR)
