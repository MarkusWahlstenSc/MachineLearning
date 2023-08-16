function model = ANNModel(TrainingData, parameters)
%%
%
%
%%

model = feedforwardnet(parameters.size');
model.trainParam.showWindow = 0;
model = train(model, TrainingData.inputs', TrainingData.outputs');

end