function model = RandomForestModel(TrainingData, parameters)
%%
%
%
%%
t     = templateTree('Reproducible', true, 'MinLeafSize', 5, 'MaxNumSplits', 5);
model = fitrensemble(TrainingData.inputs, TrainingData.outputs, ...
                     'OptimizeHyperparameters', 'auto', 'Learners', t,...
                     'HyperparameterOptimizationOptions', struct('Verbose', 0, 'ShowPlots', false));

end