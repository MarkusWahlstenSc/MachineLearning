function model = NaiveBayesModel(TrainingData, parameters)
%%
%
%
%%

model = fitcnb(TrainingData.inputs, TrainingData.outputs);

end