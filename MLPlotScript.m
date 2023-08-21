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
variables_used = [];

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
    if contains(tmp_data_field, '.')
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
        end
        hold off
    elseif strcmp(tmp_plot_option.plotType, 'histogram')
        if contains(tmp_plot_option.xAxis, '.')
            xlabel_data = eval(['MLDataSet.', tmp_plot_option.xAxis]);
        else
            xlabel_data = MLDataSet.(tmp_plot_option.xAxis);
        end
        hold on
        histogram(xlabel_data, 'BinLimits', tmp_plot_option.edges, 'NumBins', tmp_plot_option.nBins)
        hold off
    else
        disp('Plot type not recognized')
    end


    title(replace(tmp_plot_option.title, '_', '\_'))
    xlabel(replace(tmp_plot_option.xLabel, '_', '\_'))
    ylabel(replace(tmp_plot_option.yLabel, '_', '\_'))
end

end
