% MATLAB script to generate infographics for optimizer results with custom colors

close all
clear
clc

% Define the list of optimisers
optimizers = {'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Nadam', 'RMSprop', 'SGD'};

% Define directories for 'with outliers' and 'without outliers'
dirs = {'featurewith', 'featurewout'};

% Initialize data arrays to store the Average MAE and RMSE
avg_mae_with = [];
avg_rmse_with = [];
avg_mae_wout = [];
avg_rmse_wout = [];

% Initialize data arrays to store X, Y, Z MAE and RMSE
x_mae_with = [];
y_mae_with = [];
z_mae_with = [];
x_rmse_with = [];
y_rmse_with = [];
z_rmse_with = [];

x_mae_wout = [];
y_mae_wout = [];
z_mae_wout = [];
x_rmse_wout = [];
y_rmse_wout = [];
z_rmse_wout = [];

% Loop through each optimizer and grab data from .txt files
for i = 1:length(optimizers)
    opt = optimizers{i};
    
    % Loop through each directory (with outliers and without outliers)
    for j = 1:length(dirs)
        folder = dirs{j};
        txt_file = fullfile('results', opt, folder, ['mop_accuracy_', opt, '.txt']);
        
        % Read the .txt file and extract the last row (average MAE, RMSE)
        data = dlmread(txt_file);
        avg_mae = data(end, 7);  % Average MAE (7th column)
        avg_rmse = data(end, 8); % Average RMSE (8th column)
        
        % Grab X, Y, Z MAE and RMSE
        x_mae = mean(data(1:end-1, 1));  % X MAE (1st column)
        y_mae = mean(data(1:end-1, 3));  % Y MAE (3rd column)
        z_mae = mean(data(1:end-1, 5));  % Z MAE (5th column)
        x_rmse = mean(data(1:end-1, 2)); % X RMSE (2nd column)
        y_rmse = mean(data(1:end-1, 4)); % Y RMSE (4th column)
        z_rmse = mean(data(1:end-1, 6)); % Z RMSE (6th column)

        % Store the results based on the folder (with or without outliers)
        if strcmp(folder, 'featurewith')
            avg_mae_with(end+1) = avg_mae;
            avg_rmse_with(end+1) = avg_rmse;
            x_mae_with(end+1) = x_mae;
            y_mae_with(end+1) = y_mae;
            z_mae_with(end+1) = z_mae;
            x_rmse_with(end+1) = x_rmse;
            y_rmse_with(end+1) = y_rmse;
            z_rmse_with(end+1) = z_rmse;
        else
            avg_mae_wout(end+1) = avg_mae;
            avg_rmse_wout(end+1) = avg_rmse;
            x_mae_wout(end+1) = x_mae;
            y_mae_wout(end+1) = y_mae;
            z_mae_wout(end+1) = z_mae;
            x_rmse_wout(end+1) = x_rmse;
            y_rmse_wout(end+1) = y_rmse;
            z_rmse_wout(end+1) = z_rmse;
        end
    end
end

% Custom colors: blue, green, light blue, yellow, orange, red
custom_colors = [0 0.4470 0.7410;   % Blue
                 0.4660 0.6740 0.1880;   % Green
                 0.3010 0.7450 0.9330;   % Light Blue
                 0.9290 0.6940 0.1250;   % Yellow
                 0.8500 0.3250 0.0980;   % Orange
                 0.6350 0.0780 0.1840];  % Red

% --- Figure 1: Bar graph for Average MAE and RMSE for each optimizer ---
figure;
bar_data = [avg_mae_with', avg_rmse_with', avg_mae_wout', avg_rmse_wout'];
b = bar(bar_data, 'grouped');

% Set custom colors
b(1).FaceColor = custom_colors(1,:);  % Blue
b(2).FaceColor = custom_colors(2,:);  % Green
b(3).FaceColor = custom_colors(3,:);  % Light Blue
b(4).FaceColor = custom_colors(4,:);  % Yellow

% Set the x-axis labels to optimizer names (repeating for with and without)
xticks(1:length(optimizers));
xticklabels(optimizers);

% Add labels and title
title('Average MAE and RMSE for each Optimizer (With and Without Outliers)');
ylabel('Error');
legend('MAE with Outliers', 'RMSE with Outliers', 'MAE without Outliers', 'RMSE without Outliers');
grid on;

% --- Figure 2: X, Y, Z MAE and RMSE (with Outliers) ---
figure;

% Concatenate MAE and RMSE values for X, Y, Z (with outliers)
bar_data_xyz_with = [x_mae_with; y_mae_with; z_mae_with; x_rmse_with; y_rmse_with; z_rmse_with]';
b = bar(bar_data_xyz_with, 'grouped');  % Ensure 'grouped' bar plot

% Set custom colors
b(1).FaceColor = custom_colors(1,:);  % Blue (X MAE)
b(2).FaceColor = custom_colors(2,:);  % Green (Y MAE)
b(3).FaceColor = custom_colors(3,:);  % Light Blue (Z MAE)
b(4).FaceColor = custom_colors(4,:);  % Yellow (X RMSE)
b(5).FaceColor = custom_colors(5,:);  % Orange (Y RMSE)
b(6).FaceColor = custom_colors(6,:);  % Red (Z RMSE)

% Set the x-axis labels to optimizer names
xticks(1:length(optimizers));
xticklabels(optimizers);

% Add labels and title
title('X, Y, Z Error Rates with Outliers');
ylabel('Error');
legend('X MAE', 'Y MAE', 'Z MAE', 'X RMSE', 'Y RMSE', 'Z RMSE');
grid on;

% --- Figure 3: X, Y, Z MAE and RMSE (without Outliers) ---
figure;

% Concatenate MAE and RMSE values for X, Y, Z (without outliers)
bar_data_xyz_wout = [x_mae_wout; y_mae_wout; z_mae_wout; x_rmse_wout; y_rmse_wout; z_rmse_wout]';
b = bar(bar_data_xyz_wout, 'grouped');  % Ensure 'grouped' bar plot

% Set custom colors
b(1).FaceColor = custom_colors(1,:);  % Blue (X MAE)
b(2).FaceColor = custom_colors(2,:);  % Green (Y MAE)
b(3).FaceColor = custom_colors(3,:);  % Light Blue (Z MAE)
b(4).FaceColor = custom_colors(4,:);  % Yellow (X RMSE)
b(5).FaceColor = custom_colors(5,:);  % Orange (Y RMSE)
b(6).FaceColor = custom_colors(6,:);  % Red (Z RMSE)

% Set the x-axis labels to optimizer names
xticks(1:length(optimizers));
xticklabels(optimizers);

% Add labels and title
title('X, Y, Z Error Rates without Outliers');
ylabel('Error');
legend('X MAE', 'Y MAE', 'Z MAE', 'X RMSE', 'Y RMSE', 'Z RMSE');
grid on;
