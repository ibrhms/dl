import audio2mc 

filetest = "dataset//park.wav"
matfile = "dataset//park.mat"
audio2mc.createTrainingData(audioFile = filetest, labelFile = matfile)