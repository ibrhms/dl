
import librosa 
import numpy as np
import os
import shutil
import scipy.io as sio
import math
import csv

def round(y):
    y_f = math.floor(y)
    if (y -  y_f > 0.5):
        return y_f +1.0
    else:
        return y_f
        
def speech2melspec(filename, outfilename, sr_out = 8000):

    y, sr = librosa.load(filename, sr = sr_out)

    melspec =  librosa.feature.melspectrogram(y=y,sr=sr, n_mels=256)
    #print(melspec.shape)
    #print(n_fft)
    np.save(outfilename, melspec)
    
def labelDownsampling(y, numSamplesOut):
    numSamplesIn = y.shape[0]
    if numSamplesOut >= numSamplesIn:
        return None
    ratio = numSamplesIn / numSamplesOut

    result = np.zeros((numSamplesOut,1))
    idx = 0
    for i in range(0, numSamplesOut):
        idx_l = int(math.floor(idx))
        idx_h = idx_l +1
        diff = idx - idx_l
        y_l = y[idx_l]
        y_h = y[idx_h] 
        result[i] = round(y_h*diff + y_l*(1.0 - diff))
        idx += ratio
    return result
        
def createTrainingData(audioFile, labelFile, sr_out = 8000, window_size = 2048, hop_size = 512):
    audioDirName = os.path.dirname(audioFile)
    audioBaseName = os.path.basename(audioFile)
    audioNameNoExt = os.path.splitext(audioBaseName)[0]

    # labelDirName = os.path.dirname(labelFile)
    # labelBaseName = os.path.basename(labelFile)
    # labelNameNoExt = os.path.splitext(labelBaseName)[0]

    outdir = os.path.join(audioDirName, "windowed")
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    y, _ = librosa.load(audioFile, sr = sr_out)
    
    #banks
    banks = librosa.util.frame(y=y, frame_length=window_size, hop_length=hop_size)
    window = np.hanning(window_size).reshape((-1, 1))
    banks_windowed = window * banks
    num_banks = banks_windowed.shape[1]
    print("num of banks :{}".format(num_banks))
    
    #label
    labels = np.array(sio.loadmat(labelFile)['y_label'])
    labels_reduced = labelDownsampling(labels, num_banks).ravel()
    labels_dict = dict()

    for i in range(num_banks): 
        filename_no_dir = audioNameNoExt + "_" +str(i).zfill(5) + "_audio.npy"
        
        labels_dict[filename_no_dir] = int(labels_reduced[i])

        filename = os.path.join(outdir, filename_no_dir)
        np.save(filename, banks_windowed[:,i])

    w = csv.writer(open("labels.csv", "w"))
    for key, val in labels_dict.items():
        w.writerow([key, val])
    

    # np.save(labelFileName + "_label.npy", labelDownSampled)
    # melspec = np.transpose(librosa.feature.melspectrogram(y = y, sr = sr))
    # np.save(audioFileName + "_audio.npy", melspec)
    # np.save(labelFileName + "_label.npy", labelDownSampled)


