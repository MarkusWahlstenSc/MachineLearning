function MLRunScript(json_file_path)
%%
%
%
%%

[model_names, DataSettings, ModelParameters] = ReadJson(json_file_path);
[TrainingData, ValidationData, DataSet]      = GetData(DataSettings);
DataSet.predicted_output                     = GetPredictedData(model_names, ModelParameters, TrainingData, ValidationData);

SaveResultData(DataSet, DataSettings)

end

%%
function [model_names, DataSettings, ModelParameters] = ReadJson(json_file_name)

fid   = fopen(json_file_name);
raw   = fread(fid, inf);
str   = char(raw');

fclose(fid);

vals         = jsondecode(str);
model_names  = vals.modelsToRun;
model_params = vals.modelParameters;
DataSettings = vals.dataSettings;
n_models     = length(model_names);
n_params     = length(model_params);
ModelParameters = cell(n_models, 1);

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
        model = SVMModel(TrainingData, parameters);
    case 'Logistic regression'
        model = LogisticRegressionModel(TrainingData, parameters);
    case 'Naive Bayes'
        model = NaiveBayesModel(TrainingData, parameters);
    case 'ANN'
        model = ANNModel(TrainingData, parameters);
    case 'CNN'
        model = CNNModel(TrainingData, parameters);
    case 'LSTM'
        model = LSTMModel(TrainingData, parameters);
    case 'k-nearest'
        model = []; % No model
    case 'k-means'
        model = []; % No model
    case 'Decision tree'
        model = DecisionTreeModel(TrainingData, parameters);
    case 'Random forest'
        model = RandomForestModel(TrainingData, parameters);
    otherwise
        disp('Model name not recognized')
end

end

%%
function [TrainingData, ValidationData, ModifiedDataSet] = GetData(DataSettings)

if isempty(DataSettings.dataPath)

    n_samples       = 2000;
    train_ratio     = 0.9;
    data_set        = 'Sine';
    data_parameters = [10, 0.7; 7, 0.7];

    [TrainingData, ValidationData, ModifiedDataSet] = GenerateMLData(n_samples, train_ratio, data_set, data_parameters);
else
    [TrainingData, ValidationData, ModifiedDataSet] = GetTrainingAndValidationData(DataSettings);
end

end

%%
function [TrainingData, ValidationData, ModifiedDataSet] = GetTrainingAndValidationData(DataSettings)

DataSet = load(DataSettings.dataPath);

ModifiedDataSet = AddSignalDefinitions(DataSet, DataSettings.variableDefinitions);

FilteredDataSet = FilterDataSet(ModifiedDataSet, DataSettings);

Data.inputs  = GetInputsOutputs(FilteredDataSet, DataSettings.inputs);
Data.outputs = GetInputsOutputs(FilteredDataSet, DataSettings.outputs);

[TrainingData, ValidationData] = SplitTrainingAndValidationData(Data, DataSettings.trainingRatio);

ModifiedDataSet.filtered_indices   = FilteredDataSet.filtered_indices;
filtered_tmp_indices               = find(FilteredDataSet.filtered_indices == true);
validation_indices                 = false(size(FilteredDataSet.filtered_indices));
validation_indices(filtered_tmp_indices(ValidationData.indices)) = true;
ModifiedDataSet.validation_indices = validation_indices;
ModifiedDataSet.validation_input   = ValidationData.inputs;
ModifiedDataSet.validation_output  = ValidationData.outputs;

end

%%
function signal_data = GetInputsOutputs(DataSet, signals)

n_signals   = length(signals);
n_data      = length(eval(['DataSet.', signals{1}]));
signal_data = zeros(n_data, n_signals);

for k = 1:n_signals
    if contains(signals{k}, '.')
        tmp_data = eval(['DataSet.', signals{k}]);
    else
        tmp_data = DataSet.(signals{k});
    end

    if isrow(tmp_data)
        signal_data(:, k) = tmp_data';
    else
        signal_data(:, k) = tmp_data;
    end
end

end

%%
function ModifiedDataSet = AddSignalDefinitions(DataSet, variable_definitions)

ModifiedDataSet = DataSet;

if isempty(variable_definitions)
    return
end

dat_fields      = fields(DataSet);
n_dat_fields    = length(dat_fields);
variable_names  = fields(variable_definitions);
n_variable_defs = length(variable_names);

for k = 1:n_variable_defs
    variable_name     = variable_names{k};
    tmp_variable_defs = variable_definitions.(variable_name);
    split_def         = split(tmp_variable_defs, ' ');
    n_splits          = length(split_def);
    for n = 1:n_splits
        tmp_split = split_def(n);
        if contains(tmp_split, '.')
            split_struct    = split(tmp_split, '.');
            n_struct_splits = length(split_struct);
            tmp_data_set    = DataSet;
            for p = 1:n_struct_splits
                if isfield(tmp_data_set, split_struct{p})
                    tmp_data_set = tmp_data_set.(split_struct{p});
                else
                    break
                end
                if p == n_struct_splits
                    split_def{n} = ['DataSet.', tmp_split{1}];
                end
            end
        else
            for m = 1:n_dat_fields
                tmp_signal = dat_fields{m};
                if strcmp(tmp_split, tmp_signal)
                    tmp_split = ['DataSet.', tmp_signal];
                    split_def{n} = tmp_split;
                    break
                end
            end
        end
    end
    tmp_def = join(split_def);
    ModifiedDataSet.(variable_name) = eval(tmp_def{1});
end
    
end

%%
function FilteredDataSet = FilterDataSet(DataSet, DataSettings)

dat_fields    = fields(DataSet);
n_data        = length(eval(['DataSet.', DataSettings.inputs{1}]));
n_dat_fields  = length(dat_fields);
n_conditions  = length(DataSettings.conditions);
valid_indices = true(1, n_data);

for k = 1:n_conditions
    tmp_condition   = DataSettings.conditions{k};
    split_condition = split(tmp_condition, ' ');
    n_splits        = length(split_condition);
    for n = 1:n_splits
        tmp_split = split_condition(n);
        if contains(tmp_split, '.')
            split_struct    = split(tmp_split, '.');
            n_struct_splits = length(split_struct);
            tmp_data_set    = DataSet;
            for p = 1:n_struct_splits
                if isfield(tmp_data_set, split_struct{p})
                    tmp_data_set = tmp_data_set.(split_struct{p});
                else
                    break
                end
                if p == n_struct_splits
                    split_condition{n} = ['DataSet.', tmp_split{1}];
                end
            end
        else
            for m = 1:n_dat_fields
                tmp_signal = dat_fields{m};
                if strcmp(tmp_split, tmp_signal)
                    tmp_split = ['DataSet.', tmp_signal];
                    split_condition{n} = tmp_split;
                    break
                end
            end
        end
    end
    tmp_condition = join(split_condition);
    if isrow(eval(tmp_condition{1}))
        valid_indices = valid_indices & eval(tmp_condition{1});
    else
        valid_indices = valid_indices & eval(tmp_condition{1})';
    end
end

FilteredDataSet = FilterDataSetCondition(DataSet, valid_indices, DataSettings);
FilteredDataSet.filtered_indices = valid_indices;

end

%%
function FilteredDataSet = FilterDataSetCondition(DataSet, valid_indices, DataSettings)

dat_fields   = {DataSettings.inputs{:}, DataSettings.outputs{:}};
n_dat_fields = length(dat_fields);

for k = 1:n_dat_fields
    eval(['FilteredDataSet.', dat_fields{k}, ' = DataSet.', dat_fields{k}, '(valid_indices);']);
end

end

%%
function [TrainingData, ValidationData] = SplitTrainingAndValidationData(Data, training_ratio)

data_indices = 1:length(Data.inputs(:, 1));

n_total_samples = size(Data.inputs, 1);
n_verif_samples = round(n_total_samples * (1 - training_ratio));
rand_sample_seq = sort(randperm(n_total_samples, n_verif_samples));
all_indices     = 1:n_total_samples;

% Validation data
ValidationData.inputs  = Data.inputs(rand_sample_seq, :);
ValidationData.outputs = Data.outputs(rand_sample_seq);
ValidationData.indices = data_indices(rand_sample_seq);

% Training data
Data.inputs(rand_sample_seq, :) = [];
Data.outputs(rand_sample_seq)   = [];
all_indices(rand_sample_seq)    = [];
TrainingData.inputs  = Data.inputs;
TrainingData.outputs = Data.outputs;
TrainingData.indices = data_indices(all_indices);

end

%%
function SaveResultData(DataSet, DataSettings)

save(DataSettings.savedataPath, '-struct', 'DataSet')

end