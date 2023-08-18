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

dat_fields         = fields(MLDataSet);
n_fields           = length(dat_fields);
validation_indices = MLDataSet.validation_indices;
n_valid_indices    = length(validation_indices);

for k = 1:n_fields
    if (length(MLDataSet.(dat_fields{k})) == n_valid_indices)
        ConvertedMLDataSet.(dat_fields{k}) = MLDataSet.(dat_fields{k})(validation_indices);
    else
        ConvertedMLDataSet.(dat_fields{k}) = MLDataSet.(dat_fields{k});
    end
end

end

%%
function PlotData(MLDataSet, PlotOptions)

n_plots = numel(PlotOptions);

for k = 1:n_plots
    tmp_plot_option = PlotOptions(k);
    fig = figure(tmp_plot_option.figure);
    subplot(tmp_plot_option.subplot(1), tmp_plot_option.subplot(2), tmp_plot_option.subplot(3))

    yAxis = tmp_plot_option.yAxis;
    nAxis = numel(yAxis);
    hold on
    for n = 1:nAxis
        plot(MLDataSet.(tmp_plot_option.xAxis), MLDataSet.(yAxis{n})(:, 1), '*')
    end
    hold off
    title(tmp_plot_option.title)
    xlabel(tmp_plot_option.xLabel)
    ylabel(tmp_plot_option.yLabel)
end

end
