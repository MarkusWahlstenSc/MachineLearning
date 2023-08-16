function model = DecisionTreeModel(TrainingData, parameters)
%%
%
%
%%

model = fitctree(TrainingData.inputs, TrainingData.outputs);

end