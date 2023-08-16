function model = LSTMModel(TrainingData, parameters)
%%
%
%
%%

n_inputs  = size(TrainingData.inputs, 2);
n_data    = size(TrainingData.inputs, 1);
n_classes = length(unique(TrainingData.outputs));

for k = 1:n_data
    ConvertedTrainingData.inputs(k) = mat2cell(TrainingData.inputs(k, :)/20, 1, n_inputs);
end

ConvertedTrainingData.outputs = categorical(TrainingData.outputs');

init_model = [ ...
               sequenceInputLayer(1)
               lstmLayer(parameters.nHiddenUnits, 'OutputMode', 'last')
               fullyConnectedLayer(n_classes)
               softmaxLayer
               classificationLayer ...
             ];

model = trainNetwork(ConvertedTrainingData.inputs', ConvertedTrainingData.outputs', init_model, trainingOptions('sgdm', ...
                                                                                                                Plots = 'none', ...
                                                                                                                Verbose = 0));

end