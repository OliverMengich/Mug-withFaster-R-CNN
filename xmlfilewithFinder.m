Gtruth = MuGgTruth(1:144,:);

shuffleddata = randperm(height(Gtruth));
index = floor(0.6*height(Gtruth));
training = Gtruth(shuffleddata(1:index),:);
testing = Gtruth(shuffleddata(index+1:end),:);
options = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

[detector,info] = trainFasterRCNNObjectDetector( training,'resnet50',options)

videoreader = vision.VideoFileReader('MugVideoTest.mp4');
videoplayer = vision.DeployableVideoPlayer();

while ~isDone(videoreader)
    videoframe = step(videoreader);
  bbox = step(detector,videoframe);
   I = insertShape(videoframe,'rectangle',bbox);
   step(videoplayer,I);
end
