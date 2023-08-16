function model = SVMModel(TrainingData, parameters)
%%
%
%
%%

model = fitcsvm(TrainingData.inputs, TrainingData.outputs, 'Standardize', true, ...
               'KernelFunction', parameters.kernelfunction);
%           model = fitcsvm(input_data',output_data','Standardize',true);
%    case 'logistic regression'
%        [model, ~] = glmfit(input_data',[output_data' ones(size(output_data'))], ...
%                            'binomial', 'logit');
%    case 'feedforward'
%        model_size = [15, 10, 5];
%        model = feedforwardnet(model_size);
%        model = train(model, input_data, output_data);

end