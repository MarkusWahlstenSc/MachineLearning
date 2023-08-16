function model = LogisticRegressionModel(TrainingData, parameters)
%%
%
%
%%

model = glmfit(TrainingData.inputs, TrainingData.outputs,'binomial','link','logit');

end