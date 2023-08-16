function MLRunScript(json_file_path)
%%
%
%
%%

[model_names, DataSettings, ModelParameters] = ReadJson(json_file_path);
[TrainingData, ValidationData]       = GetData(DataSettings);
predicted_data                       = GetPredictedData(model_names, ModelParameters, TrainingData, ValidationData);

end

%%
function [model_names, DataSettings, ModelParameters] = ReadJson(setup_file_name)

fname = setup_file_name;
fid   = fopen(fname);
raw   = fread(fid, inf);
str   = char(raw');

fclose(fid);

vals         = jsondecode(str);
model_names  = vals.modelsToRun;
model_params = vals.modelParameters;
DataSettings = vals.dataSettings;
n_models     = length(model_names);
n_params     = length(model_params);
ModelParameters   = cell(n_models, 1);

for k = 1:n_models
    tmp_model_names = model_names(k);
    for m = 1:n_params
        tmp_parameter = model_params{m};
        if strcmp(tmp_parameter.model, tmp_model_names)
            ModelParameters{k} = tmp_parameter;
        end
    end
end

end

%%
function predicted_output = GetPredictedData(model_names, model_parameters, TrainingData, ValidationData)

n_models         = length(model_names);
predicted_output = zeros(length(ValidationData.outputs), n_models);
accuracy         = zeros(n_models);
max_str_len      = 19;
disp('----- Accuracy -----')
for k = 1:n_models
    model_name = model_names{k};
    parameters = model_parameters{k};
    model                  = TrainModel(TrainingData, parameters, model_name);
    predicted_output(:, k) = PredictOutput(model, model_name, ValidationData, TrainingData, parameters);
    accuracy(k)            = 1 - sum(abs(predicted_output(:, k) - ValidationData.outputs)) / length(ValidationData.outputs);
    whitespaces            = char(repmat({' '}, max_str_len - length(model_name), 1))';
    disp([model_name, ': ', whitespaces, num2str(accuracy(k) * 100), '%'])
end

end

%%
function predicted_output = PredictOutput(model, model_name, ValidationData, TrainingData, parameters)

switch model_name
    case 'SVM'
        predicted_output = predict(model, ValidationData.inputs);
    case 'Logistic regression'
        predicted_output = glmval(model, ValidationData.inputs, 'logit') > 0.5;
    case 'Naive Bayes'
        predicted_output = predict(model, ValidationData.inputs);
    case 'ANN'
        predicted_output = model(ValidationData.inputs') > 0.5;
    case 'CNN'      
        ConvertedValidationData = ConvertDataToCells(ValidationData.inputs);
        predicted_output        = double(model.classify(ConvertedValidationData.inputs')) - 1;
    case 'LSTM'
        ConvertedValidationData = ConvertDataToCells(ValidationData.inputs);
        predicted_output        = double(model.classify(ConvertedValidationData.inputs')) - 1;
    case 'k-nearest'
        output_index     = knnsearch(TrainingData.inputs, ValidationData.inputs, 'Distance', parameters.distanceMetric);
        predicted_output = TrainingData.outputs(output_index);
    case 'k-means'
        predicted_output = kmeans(ValidationData.inputs, 2) - 1;
        if sum(abs(predicted_output - ValidationData.outputs)) / length(predicted_output) > 0.5
            predicted_output = double(predicted_output == 0);
        end
    case 'Decision tree'
        predicted_output = predict(model, ValidationData.inputs);
    case 'Random forest'
        predicted_output = predict(model, ValidationData.inputs) > 0.5;
    otherwise
        disp('Model name not recognized')
end

end

%%
function converted_data = ConvertDataToCells(data)

[n_data, n_inputs] = size(data);
    
for k = 1:n_data
    converted_data.inputs(k) = mat2cell(data(k, :)/20, 1, n_inputs);
end

end

%%
function model = TrainModel(TrainingData, parameters, model_name)

switch model_name
    case 'SVM'
        model                  = SVMModel(TrainingData, parameters);
    case 'Logistic regression'
        model                  = LogisticRegressionModel(TrainingData, parameters);
    case 'Naive Bayes'
        model                  = NaiveBayesModel(TrainingData, parameters);
    case 'ANN'
        model                  = ANNModel(TrainingData, parameters);
    case 'CNN'
        model                  = CNNModel(TrainingData, parameters);
    case 'LSTM'
        model                  = LSTMModel(TrainingData, parameters);
    case 'k-nearest'
        model                  = []; % No model
    case 'k-means'
        model                  = []; % No model
    case 'Decision tree'
        model                  = DecisionTreeModel(TrainingData, parameters);
    case 'Random forest'
        model                  = RandomForestModel(TrainingData, parameters);
    otherwise
        disp('Model name not recognized')
end

end

%%
function [TrainingData, ValidationData] = GetData(DataSettings)

if isempty(DataSettings.dataPath)

    n_samples       = 2000;
    train_ratio     = 0.9;
    data_set        = 'Sine';
    data_parameters = [10, 0.7; 7, 0.7];

    [TrainingData, ValidationData] = GenerateMLData(n_samples, train_ratio, data_set, data_parameters);
else
    [TrainingData, ValidationData] = GetTrainingAndValidationData(DataSettings);
end

end

%%
function [TrainingData, ValidationData] = GetTrainingAndValidationData(DataSettings)

DataSet = load(DataSettings.dataPath);

ModifiedDataSet = AddSignalDefinitions(DataSet, DataSettings.variableDefinitions);

FilteredDataSet = FilterDataSet(ModifiedDataSet, DataSettings.conditions);

Data.inputs  = GetInputsOutputs(FilteredDataSet, DataSettings.inputs);
Data.outputs = GetInputsOutputs(FilteredDataSet, DataSettings.outputs);

[TrainingData, ValidationData] = SplitTrainingAndValidationData(Data, DataSettings.trainingRatio);

end

%%
function signal_data = GetInputsOutputs(DataSet, signals)

n_signals   = length(signals);
n_data      = length(DataSet.(signals{1}));
signal_data = zeros(n_data, n_signals);

for k = 1:n_signals
    if isrow(DataSet.(signals{k}))
        signal_data(:, k) = DataSet.(signals{k})';
    else
        signal_data(:, k) = DataSet.(signals{k});
    end
end

end

%%
function ModifiedDataSet = AddSignalDefinitions(DataSet, variable_definitions)

ModifiedDataSet = DataSet;

end

%%
function FilteredDataSet = FilterDataSet(DataSet, conditions)

dat_fields    = fields(DataSet);
n_data        = length(DataSet.([dat_fields{1}]));
n_dat_fields  = length(dat_fields);
n_conditions  = length(conditions);
valid_indices = true(1, n_data);

for k = 1:n_conditions
    tmp_condition   = conditions{k};
    split_condition = split(tmp_condition, ' ');
    n_splits        = length(split_condition);
    for n = 1:n_splits
        tmp_split = split_condition(n);
        for m = 1:n_dat_fields
            tmp_signal = dat_fields{m};
            if strcmp(tmp_split, tmp_signal)
                tmp_split = ['DataSet.', tmp_signal];
                split_condition{n} = tmp_split;
                break
            end
        end
    end
    tmp_condition = join(split_condition);
    valid_indices = valid_indices & eval(tmp_condition{1});
end

FilteredDataSet = FilterDataSetCondition(DataSet, valid_indices);

end

%%
function FilteredDataSet = FilterDataSetCondition(DataSet, valid_indices)

dat_fields   = fields(DataSet);
n_dat_fields = length(dat_fields);

for k = 1:n_dat_fields
    FilteredDataSet.(dat_fields{k}) = DataSet.(dat_fields{k})(valid_indices);
end

end

%%
function [TrainingData, ValidationData] = SplitTrainingAndValidationData(Data, training_ratio)

data_indices = 1:length(Data.inputs(:, 1));

n_total_samples = size(Data.inputs, 1);
n_verif_samples = round(n_total_samples * (1 - training_ratio));
rand_sample_seq = randperm(n_total_samples, n_verif_samples);
all_indices     = 1:n_total_samples;

% Validation data
ValidationData.inputs  = Data.inputs(rand_sample_seq, :);
ValidationData.outputs = Data.outputs(rand_sample_seq);
ValidationData.indices = data_indices(rand_sample_seq);

% Training data
Data.inputs(rand_sample_seq, :) = [];
Data.outputs(rand_sample_seq)   = [];
all_indices(rand_sample_seq)   = [];
TrainingData.inputs  = Data.inputs;
TrainingData.outputs = Data.outputs;
TrainingData.indices = data_indices(all_indices);

end