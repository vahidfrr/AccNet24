% Specify the root for the training folder
rootFolder = fullfile('D:\Vahid\FINAL FILES\Usman_2\train');

%Specify categoties in the training folder
categories  = {'4MVPA','3LPA','2Sed','1Sleep'};

%read image data
train = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');  
Input for Validation
% Specify the root for the validation folder
rootFolder = fullfile('D:\Vahid\FINAL FILES\Usman_2\val');

%Specify categoties in the validation folder
categories  = {'4MVPA','3LPA','2Sed','1Sleep'};

%read image data
validation = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames'); 
Input for Testing
% Specify the root for the test folder
rootFolder = fullfile('D:\Vahid\FINAL FILES\Usman_2\test');

% Specify the categories for the test folder
categories  = {'4MVPA','3LPA','2Sed','1Sleep'};

%read image data
test = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders',true, ...
 'LabelSource','foldernames'); 

Labels extraction
trainingLabels = train.Labels;
validationLabels = validation.Labels;
testlabel = test.Labels;
Training and validating
Fine-Tunning for Resnet-101 model
net = resnet101;
net.Layers(1)
inputSize = net.Layers(1).InputSize;
 lgraph = layerGraph(net);
% 
% % Remove the the last 3 layers from ResNet-101. 
 layersToRemove = {
     'fc1000'
     'prob'
     'ClassificationLayer_predictions'
     };
 lgraph = removeLayers(lgraph, layersToRemove);
% Specify the number of classes the network should classify.
 numClasses = 4;
% % Define new classification layers.
 newLayers = [
     fullyConnectedLayer(numClasses, 'Name', 'rcnnFC')
     softmaxLayer('Name', 'rcnnSoftmax')
     classificationLayer('Name', 'rcnnClassification')
     ];
 lgraph = addLayers(lgraph, newLayers);
 lgraph = connectLayers(lgraph,  'pool5' , 'rcnnFC');

% Resize images using augmentedImageDatastore function for Resnet model
train = augmentedImageDatastore(inputSize(1:2),train);
validation = augmentedImageDatastore(inputSize(1:2),validation);
test = augmentedImageDatastore(inputSize(1:2),test);

% set the parameters
miniBatchSize = 16;
options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ... 
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',0.0001, ...
    'Shuffle','every-epoch', ...   
    'ValidationData',validation, ...
   'ValidationFrequency',30, ...
    'Verbose',false, ...
      'Plots','training-progres', ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));


disp ('ResNet training started')
% start training 
ResNet_FineTuned = trainNetwork(train,lgraph,options);

% Compute performance based on CNN without BiLSTM

[YPred,scores] = classify(ResNet_FineTuned,test);
accuracy_CNN = mean(YPred == testlabel)

RICA preprocessing for features dimensionality

% Features extraction based on Fine-Tuned CNN (net2)

featureLayer =   'pool5';
trainingFeatures_ResNet = activations(ResNet_FineTuned,train,featureLayer, ...
    'MiniBatchSize', miniBatchSize, 'OutputAs', 'columns');

validationFeatures_ResNet = activations(ResNet_FineTuned,validation,featureLayer, ...
    'MiniBatchSize', miniBatchSize, 'OutputAs', 'columns');

testFeatures_ResNet = activations(ResNet_FineTuned,test, featureLayer, ...
    'MiniBatchSize', miniBatchSize, 'OutputAs', 'columns');

% features input for RICA
disp ('RICA training started')

rng default % For reproducibility
q = 100; % number of features obtained from RICA
Mdl = rica(trainingFeatures_ResNet',q,'IterationLimit',80);
trainingFeatures_RICA = transform(Mdl,trainingFeatures_ResNet');
validationFeatures_RICA =  transform(Mdl,validationFeatures_ResNet');
testFeatures_RICA = transform(Mdl,testFeatures_ResNet');

BiLSTM training

rng(13);

trainingFeatures_RICA_Cell = {};
trainingFeatures_RICA_Cell{end+1} =    trainingFeatures_RICA';

trainLables_Cell = {};
trainLables_Cell{end+1} = trainingLabels';

validationFeatures_RICA_Cell = {};
validationFeatures_RICA_Cell{end+1} =  validationFeatures_RICA';
% 
validationLabels_Cell = {};
validationLabels_Cell{end+1} = validationLabels';


numFeatures =100;
numHiddenUnits = 20;
numClasses = 4;
layers = [ ...
    sequenceInputLayer(numFeatures)
         bilstmLayer(numHiddenUnits,'OutputMode','sequence','RecurrentWeightsInitializer', 'he')
     fullyConnectedLayer(numClasses, 'WeightsInitializer','he')
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ... 
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',1500, ...
    'ValidationData',{validationFeatures_RICA_Cell,validationLabels_Cell}, ...
    'ValidationFrequency',30, ...
    'SequenceLength','longest', ...
    'Plots','training-progress',...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));


disp ('BiLSTM training started')
%Train lstm
lstm = trainNetwork(trainingFeatures_RICA_Cell',trainLables_Cell,layers,options);

Predict validation set

[predictedLabels_validation, devlp_scores_validation] = classify(lstm,validationFeatures_RICA');

% Plot confusion matrix
plotconfusion(validationLabels',predictedLabels_validation)

Predict testset

[predictedLabels_test, devlp_scores_test] = classify(lstm,testFeatures_RICA');

% Plot confusion matrix
plotconfusion(testlabel',predictedLabels_test)

Perfirmance evaluation on the validation set (voting per 3 image)

% There are three images for each window, since there were 3 axes (i.e., x, y, z).
% We do majority voting for every 3 images that belong to the same window


predictedLabels_validationset_1PerCateg=[];
for i = 1:3:length(predictedLabels_validation)
    predictedLabels_validationset_1PerCateg = [predictedLabels_validationset_1PerCateg, majorityvote(predictedLabels_validation (i:i+2))];  
end
% We also do take one of the true labeles to create the vectors, which
% shows th correct lables for each window.

validationlabel_1PerCateg=[];
for i = 1:3:length(validationLabels)
    validationlabel_1PerCateg = [validationlabel_1PerCateg, validationLabels(i)];  
end

% Plot confusion matrix 
plotconfusion(validationlabel_1PerCateg',predictedLabels_validationset_1PerCateg');

Perfirmance evaluation on the testset (voting per 3 image)

% There are three images for each window, since there were 3 axes (i.e., x, y, z).
% We do majority voting for every 3 images that belong to the same window


predictedLabels_testset_1PerCateg=[];
for i = 1:3:length(predictedLabels_test)
    predictedLabels_testset_1PerCateg = [predictedLabels_testset_1PerCateg, majorityvote(predictedLabels_test (i:i+2))];  
end
% We also do take one of the true labeles to create the vectors, which
% shows th correct lables for each window.

testlabel_1PerCateg=[];
for i = 1:3:length(testlabel)
    testlabel_1PerCateg = [testlabel_1PerCateg, testlabel(i)];  
end



% Plot confusion matrix
plotconfusion(testlabel_1PerCateg',predictedLabels_testset_1PerCateg');


