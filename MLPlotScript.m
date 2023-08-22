function MLPlotScript(json_file_path)
%%
%
%
%%

[model_names, PlotOptions, dataPath] = ReadJson(json_file_path);
MLDataSet                            = GenerateMLResult(dataPath, PlotOptions);
PlotData(MLDataSet, PlotOptions);

end

%%
function [model_names, PlotOptions, dataPath] = ReadJson(json_file_name)

fid   = fopen(json_file_name);
raw   = fread(fid, inf);
str   = char(raw');

fclose(fid);

vals        = jsondecode(str);
model_names = vals.modelsToPlot;
PlotOptions = vals.PlotOptions;
dataPath    = vals.dataPath;
n_models    = length(model_names);

end

%%
function variables_used = GetVariableUsedList(PlotOptions)

n_fields = length(PlotOptions);
variables_used = {'validation_output'};

for k = 1:n_fields
    tmp_plot_options = PlotOptions{k};

    % Add xAxis variables
    if ~any(strcmp(variables_used, tmp_plot_options.xAxis))
        variables_used{end + 1} = tmp_plot_options.xAxis;
    end

    % Add yAxis variables
    n_yAxis = numel(tmp_plot_options.yAxis);
    for n = 1:n_yAxis
        if ~any(strcmp(variables_used, tmp_plot_options.yAxis{n}))
            variables_used{end + 1} = tmp_plot_options.yAxis{n};
        end
    end
end

end

%%
function MLDataSet = GenerateMLResult(dataPath, PlotOptions)

MLDataSet = load(dataPath);

MLDataSet.predicted_accuracy = ComputeAccuracy(MLDataSet);
MLDataSet                    = ConvertPlotData(MLDataSet, PlotOptions);

end

%%
function ml_accuracy = ComputeAccuracy(DataSet)

ml_accuracy = 1 - sum(abs(DataSet.predicted_output - DataSet.validation_output)) / length(DataSet.validation_output);

end

%%
function ConvertedMLDataSet = ConvertPlotData(MLDataSet, PlotOptions)

dat_fields         = GetVariableUsedList(PlotOptions);
n_fields           = length(dat_fields);
validation_indices = MLDataSet.validation_indices;
n_valid_indices    = length(validation_indices);

for k = 1:n_fields

    tmp_data_field = dat_fields{k};
    
    if strcmp(tmp_data_field, 'predicted_accuracy')   || ...
       strcmp(tmp_data_field, 'predicted_accuracy_0') || ...
       strcmp(tmp_data_field, 'predicted_accuracy_1')
        continue
    elseif contains(tmp_data_field, '.')
        n_tmp_data = eval(['length(MLDataSet.', dat_fields{k}, ')']);
        if (n_tmp_data == n_valid_indices)
            eval(['ConvertedMLDataSet.', dat_fields{k}, ' = MLDataSet.', dat_fields{k}, '(validation_indices);']);
        else
            eval(['ConvertedMLDataSet.', dat_fields{k}, ' = MLDataSet.', dat_fields{k}, ';']);
        end

    else
        n_tmp_data = length(MLDataSet.(dat_fields{k}));
        if (n_tmp_data == n_valid_indices)
            ConvertedMLDataSet.(dat_fields{k}) = MLDataSet.(dat_fields{k})(validation_indices);
        else
            ConvertedMLDataSet.(dat_fields{k}) = MLDataSet.(dat_fields{k});
        end
    end

end

end

%%
function PlotData(MLDataSet, PlotOptions)

n_plots = numel(PlotOptions);

for k = 1:n_plots
    tmp_plot_option = PlotOptions{k};
    fig = figure(tmp_plot_option.figure);
    subplot(tmp_plot_option.subplot(1), tmp_plot_option.subplot(2), tmp_plot_option.subplot(3))
    legend_text = [];
    if strcmp(tmp_plot_option.plotType, 'plot')
        yAxis = tmp_plot_option.yAxis;
        nAxis = numel(yAxis);

        if contains(tmp_plot_option.xAxis, '.')
            xlabel_data = eval(['MLDataSet.', tmp_plot_option.xAxis]);
        else
            xlabel_data = MLDataSet.(tmp_plot_option.xAxis);
        end
        hold on
        for n = 1:nAxis
            if contains(yAxis{n}, '.')
                ylabel_data = eval(['MLDataSet.', yAxis{n}, '(:, 1)''']);
            else
                ylabel_data = MLDataSet.(yAxis{n})(:, 1)';
            end
            plot(xlabel_data, ylabel_data, '*')
            legend_text{end + 1} = replace(yAxis{n}, '_', '\_');
        end
        hold off
    elseif strcmp(tmp_plot_option.plotType, 'histogram')
        if contains(tmp_plot_option.xAxis, '.')
            xlabel_data = eval(['MLDataSet.', tmp_plot_option.xAxis]);
        elseif strcmp(tmp_plot_option.xAxis, 'predicted_output')
            xlabel_data = MLDataSet.(tmp_plot_option.xAxis)(:, 1);
        else
            xlabel_data = MLDataSet.(tmp_plot_option.xAxis);
        end
        
        histogram(xlabel_data, 'BinLimits', tmp_plot_option.edges, 'NumBins', tmp_plot_option.nBins)
        legend_text = replace(tmp_plot_option.yLabel, '_', '\_');
    elseif strcmp(tmp_plot_option.plotType, 'plot bin')
        yAxis = tmp_plot_option.yAxis;
        nAxis = numel(yAxis);

        if contains(tmp_plot_option.xAxis, '.')
            xlabel_data = eval(['MLDataSet.', tmp_plot_option.xAxis]);
        else
            xlabel_data = MLDataSet.(tmp_plot_option.xAxis);
        end
        
        [xlabel_data_bin, xlabel_data_edges] = discretize(xlabel_data, linspace(tmp_plot_option.edges(1), ...
                                                           tmp_plot_option.edges(2), ...
                                                           tmp_plot_option.nBins));
        n_bins = length(xlabel_data_edges);

        for n = 1:nAxis
            hold on
            if contains(yAxis{n}, '.')
                ylabel_data(n, :) = eval(['MLDataSet.', yAxis{n}, '(:, 1)''']);
                ylabel_data_binned = BinData(ylabel_data, xlabel_data_bin);
            elseif strcmp(yAxis{n}, 'predicted_accuracy')   || ...
                   strcmp(yAxis{n}, 'predicted_accuracy_0') || ...
                   strcmp(yAxis{n}, 'predicted_accuracy_1')
                ylabel_data_binned = ComputeBinAccuracy(MLDataSet, yAxis{n}, xlabel_data_bin, n_bins);
            else
                ylabel_data(n, :) = MLDataSet.(yAxis{n})(:, 1)';
                ylabel_data_binned = BinData(ylabel_data, xlabel_data_bin);
            end

            plot(xlabel_data_edges, ylabel_data_binned)
            hold off
            legend_text{end + 1} = replace(yAxis{n}, '_', '\_');
        end        
    else
        disp('Plot type not recognized')
    end


    title(replace(tmp_plot_option.title,   '_', '\_'))
    xlabel(replace(tmp_plot_option.xLabel, '_', '\_'))
    ylabel(replace(tmp_plot_option.yLabel, '_', '\_'))
    if isfield(tmp_plot_option, 'yLimit')
        ylim(tmp_plot_option.yLimit)
    end
    legend(legend_text)
end

end

%%
function binned_data = BinData(data, data_bin)

n_bins      = max(data_bin);
binned_data = nan(1, n_bins);

for k = 1:n_bins



end

end

%%
function accuracy_data = ComputeBinAccuracy(DataSet, data_label, xlabel_data_bin, n_bins)

accuracy_data = nan(1, n_bins);

if strcmp(data_label, 'predicted_accuracy_0')
    valid_data = DataSet.validation_output == 0;
elseif strcmp(data_label, 'predicted_accuracy_1')
    valid_data = DataSet.validation_output == 1;
else
    valid_data = true(size(DataSet.validation_output));
end

xlabel_data_bin_valid = xlabel_data_bin(valid_data);
valid_output          = DataSet.validation_output(valid_data);
predicted_output      = DataSet.predicted_output(valid_data);

for k = 1:n_bins
    if ~isempty(valid_output(xlabel_data_bin_valid == k))
        accuracy_data(k) = 1 - sum(abs(valid_output(xlabel_data_bin_valid == k) - predicted_output(xlabel_data_bin_valid == k))) / length(valid_output(xlabel_data_bin_valid == k));
    end
end

end