function [TrainingData, ValidationData] = GenerateMLData(n_samples, train_fraction, data_set, data_parameters)
%%
%
%
%%


if strcmp(data_set, 'Sine')
    [TrainingData, ValidationData] = GenerateSineMLData(n_samples, train_fraction, data_parameters);
else
    disp('Not a valid data set')
    TrainingData   = [];
    ValidationData = [];
end

end

%%
function [TrainingData, ValidationData] = GenerateSineMLData(n_samples, train_fraction, data_parameters)

n_train_data = round(n_samples * train_fraction); 
n_eval_data  = n_samples - n_train_data;

TrainingData   = GenerateSineData(n_train_data, data_parameters);
ValidationData = GenerateSineData(n_eval_data,  data_parameters);

end

%%
function DataSet = GenerateSineData(n_data, data_parameters)

n_size      = 10;
n_samples   = [ceil(n_data/2), floor(n_data/2)];
data_input  = [];
data_output = [];

for k = 1:2
    mu          = data_parameters(k, 1);
    sigma       = data_parameters(k, 2);
    rand_data   = mu + sigma/10 .* randn(n_samples(k), 1) + sigma .* randn(n_samples(k), 1) .* ones(n_samples(k), 1);
    angles      = linspace(0, 2*pi, n_size) .* ones(n_samples(k), n_size);
    data_input  = [data_input;  rand_data .* sin(angles) + sigma/10 .* randn(n_samples(k), n_size)];
    data_output = [data_output; (k - 1) * ones(n_samples(k), 1)];
end

DataSet.inputs  = data_input;
DataSet.outputs = data_output;

end