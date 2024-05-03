%%  Deep Learning Network - Estimating New Requests
%% Getting Set Up
% Perform a workspace and figure reset

clear all; close all;
%% 
% Set the number of concantenated random episodes that make up the database:

no_it = 1e3; 
%% 
% Set the considered contention resolution policy, the type of DNN output and 
% whether the contention resolution policy is included as DNN input. This is indicated 
% using either a number of a letter as introduced at the beginning of this script.

type = 1; % P_ACB
no_layers = 2; % No. of hidden NN layers
%% 
% From the previous information generate the database's file name that will 
% be loaded and used to train and test the DNN.

% The obtained results will be stored in a file named 'FileName_fitrnet':
FileName_fitrnet = strcat('trained_net_no_it_',num2str(no_it),'_',num2str(type),...
    '_',num2str(no_layers));
% The loaded file containing the generated database is named
% 'FileName_database':
FileName_database = strcat('artificial_db_no_it_',num2str(no_it),'_',num2str(type));
load(FileName_database);
%% Create Array of Layers

if no_layers == 1

% Two-output network; 1 regression layer + 1 intermediate 10-neuron layer
% ******************************************
    layers = [
        featureInputLayer(5,"Name","featureinput")
        fullyConnectedLayer(10,"Name","fc_1")
        reluLayer("Name","relu_1")
        fullyConnectedLayer(2,"Name","fc_2")
        regressionLayer("Name","regressionoutput")];

elseif no_layers == 2

% Two-output network; 1 regression layer + 2 intermediate 10-neuron layers
% ******************************************

    layers = [
        featureInputLayer(5,"Name","featureinput")
        fullyConnectedLayer(10,"Name","fc_1")
        reluLayer("Name","relu_1")
        fullyConnectedLayer(10,"Name","fc_2")
        reluLayer("Name","relu_2")
        fullyConnectedLayer(2,"Name","fc_3")
        regressionLayer("Name","regressionoutput")];

elseif no_layers == 3

% Two-output network; 1 regression layer + 3 intermediate 10-neuron layers
% ******************************************

    layers = [
        featureInputLayer(5,"Name","featureinput")
        fullyConnectedLayer(10,"Name","fc_1")
        reluLayer("Name","relu_1")
        fullyConnectedLayer(10,"Name","fc_2")
        reluLayer("Name","relu_2")
        fullyConnectedLayer(10,"Name","fc_3")
        reluLayer("Name","relu_3")
        fullyConnectedLayer(2,"Name","fc_4")
        regressionLayer("Name","regressionoutput")];

elseif no_layers == 4

% Two-output network; 5-4-3-2
    layers = [
        featureInputLayer(5,"Name","featureinput")
        reluLayer("Name","relu_in")
        fullyConnectedLayer(4,"Name","fc_1")
        reluLayer("Name","relu_1")
        fullyConnectedLayer(3,"Name","fc_2")
        reluLayer("Name","relu_2")
        fullyConnectedLayer(2,"Name","fc_3")
        reluLayer("Name","relu_3")
        regressionLayer("Name","regressionoutput")];

elseif no_layers == 5

% Two-output network; 5-10-4-2
    layers = [
        featureInputLayer(5,"Name","featureinput")
        reluLayer("Name","relu_in")
        fullyConnectedLayer(10,"Name","fc_1")
        reluLayer("Name","relu_1")
        fullyConnectedLayer(4,"Name","fc_2")
        reluLayer("Name","relu_2")
        fullyConnectedLayer(2,"Name","fc_3")
        reluLayer("Name","relu_3")
        regressionLayer("Name","regressionoutput")];

elseif no_layers == 6
    % Two-output network; 5-10-8-6-4-2
    layers = [
        featureInputLayer(5,"Name","featureinput")
        reluLayer("Name","relu_in")
        fullyConnectedLayer(10,"Name","fc_1")
        reluLayer("Name","relu_1")
        fullyConnectedLayer(8,"Name","fc_2")
        reluLayer("Name","relu_2")
        fullyConnectedLayer(6,"Name","fc_3")
        reluLayer("Name","relu_3")
        fullyConnectedLayer(4,"Name","fc_4")
        reluLayer("Name","relu_4")
        fullyConnectedLayer(2,"Name","fc_5")
        reluLayer("Name","relu_5")
        regressionLayer("Name","regressionoutput")];

elseif no_layers == 7
    % Two-output network; 5-10-5-2
    layers = [
        featureInputLayer(5,"Name","featureinput")
        reluLayer("Name","relu_in")
        fullyConnectedLayer(10,"Name","fc_1")
        reluLayer("Name","relu_1")
        fullyConnectedLayer(5,"Name","fc_2")
        reluLayer("Name","relu_2")
        fullyConnectedLayer(2,"Name","fc_3")
        reluLayer("Name","relu_3")
        regressionLayer("Name","regressionoutput")];

elseif no_layers == 8
    % Two-output network; 5-10-10-4-2
    layers = [
        featureInputLayer(5,"Name","featureinput")
        reluLayer("Name","relu_in")
        fullyConnectedLayer(10,"Name","fc_1")
        reluLayer("Name","relu_1")
        fullyConnectedLayer(10,"Name","fc_2")
        reluLayer("Name","relu_2")
        fullyConnectedLayer(4,"Name","fc_3")
        reluLayer("Name","relu_3")
        fullyConnectedLayer(2,"Name","fc_4")
        reluLayer("Name","relu_4")
        regressionLayer("Name","regressionoutput")];

end


% Two-output network; 2 regression layers (NOT WORKING)
% ******************************************
%{
% creating the layer graph variable containing the network layers
lgraph = layerGraph();

% adding network branches to the layer graph
tempLayers = [
    featureInputLayer(5,"Name","featureinput")
    fullyConnectedLayer(10,"Name","fc_1")
    reluLayer("Name","relu")
    fullyConnectedLayer(2,"Name","fc_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = regressionLayer("Name","regressionoutput_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = regressionLayer("Name","regressionoutput_1");
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

% connecting layer branches:
lgraph = connectLayers(lgraph,"fc_2","regressionoutput_2");
lgraph = connectLayers(lgraph,"fc_2","regressionoutput_1");
%}
%% Plot Layers

plot(layerGraph(layers));
%% Training the Network
% Database random partition to set train and test batches:

% Skip unwanted features and re-order (last columns are response features)
data = [data(:,1:2) data(:,4:5) data(:,8) data(:,6:7)];

% Cross validation (train: 70%, test: 30%)
cv = cvpartition(size(data,1),'HoldOut',0.3); % define a random partition on the dataset
idx = cv.test;

% Separate to training and test data
dataTrain = data(~idx,:);
dataValidation  = data(idx,:);


% Cross-validation based on testing on a single episode 
% (1 episode = one peak of new access requests)
% ** DEACTIVATED ** - random permutation of indices is better (WHY IS IT NOT?)
%it_part_train = floor(0.7*no_it); it_part_test = ceil(0.3*no_it);
%it_part_train = floor(0.95*no_it); it_part_test = ceil(0.05*no_it);
%dataTrain = data(1:(it_part_train-1)*env5GConst.N_steps,:);
%dataValidation = data((it_part_test-1)*env5GConst.N_steps:end,:);

% Set the training partition
XTrain = dataTrain(:,1:5); % get the first five features as input data
YTrain = dataTrain(:,6:7); % set 'n1' and 'n2' as responses

% Set the validation partition
XValidation = dataValidation(:,1:5); % get the first five features as input data
YValidation = dataValidation(:,6:7); % set 'n1' and 'n2' as responses
%% 
% Set training options

ValidationData_cellarray = {XValidation,YValidation};

no_it_per_epoch = 1367;
val_freq = 20;

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'ValidationData',ValidationData_cellarray, ...
    'ValidationFrequency',val_freq, ...
    'ValidationPatience',floor(no_it_per_epoch/val_freq), ...
    'MaxEpochs',30, ...
    'Verbose',1);
% The 'ValidationPatience' feature is used to stop training automatically 
% when the validation loss stops decreasing. In this case, if the loss on
% the validation data does not change in 1 epoch, training is stopped.

% Loss = MSE when comparing predicted and actual response features
%% 
% Train the network:

[net, info] = trainNetwork(XTrain,YTrain,layers,options);
%% 
% Compute the validation RMSE:

% ** TEMPORARY CODE LINE WHEN USING UNUPDATED TRAINED NET RESULTS **
%XValidation = XTest; YValidation = YTest;

%%
YValidPred = predict(net,XValidation);

format long;
validation_rmse = sum(sqrt(mean((YValidation - YValidPred).^2,1)))
%% Testing the net's performance on unseen data (test dataset)
% Load the test dataset:

no_it_test = 300;
FileNameTest = strcat('artificial_db_no_it_',num2str(no_it_test),'_',num2str(type),'_TEST');
dataTest = load(FileNameTest,'data');
dataTest = dataTest.data;

dataTest = [dataTest(:,1:2) dataTest(:,4:5) dataTest(:,8) dataTest(:,6:7)];

XTest = dataTest(:,1:5); % get the first five features as input data
YTest = dataTest(:,6:7); % set 'n1' and 'n2' as responses
%% 
% Test the network's performance by measuring prediction accuracy:

YTestPred = predict(net,XTest);
%% 
% Evaluate the performance of the model by calculating the root-mean-square 
% error (RMSE) of the predicted and actual number of newly generated access requests:

format long;
test_rmse_n1 = sqrt(mean((YTest(:,1) - YTestPred(:,1)).^2,1))
test_rmse_n2 = sqrt(mean((YTest(:,2) - YTestPred(:,2)).^2,1))

test_rmse = sum(sqrt(mean((YTest - YTestPred).^2,1)))
%% 
% Check how the net performs on different episodes

YActual_plot_n1 = zeros(no_it_test,env5GConst.N_steps);
YActual_plot_n2 = zeros(no_it_test,env5GConst.N_steps);

YPred_plot_n1 = zeros(no_it_test,env5GConst.N_steps);
YPred_plot_n2 = zeros(no_it_test,env5GConst.N_steps);

YError_plot_n1 = zeros(3,env5GConst.N_steps);
YError_plot_n2 = zeros(3,env5GConst.N_steps);

for it_part = 1:1:no_it_test
    %it_part = 200; % set iteration partition to visualize performance of the trained model
    dataActual_plot = dataTest(1+(it_part-1)*env5GConst.N_steps:it_part*env5GConst.N_steps,:);

    YActual_plot_n1(it_part,:) = dataActual_plot(:,6)';
    YActual_plot_n2(it_part,:) = dataActual_plot(:,7)';

    YPred_plot = predict(net,dataActual_plot(:,1:5));
    YPred_plot_n1(it_part,:) = YPred_plot(:,1)'; YPred_plot_n2(it_part,:) = YPred_plot(:,2)';
end

YActual_plot_n1_avg = mean(YActual_plot_n1,1); YActual_plot_n2_avg = mean(YActual_plot_n2,1);
YPred_plot_n1 = mean(YPred_plot_n1,1); YPred_plot_n2 = mean(YPred_plot_n2,1);


for ii = 1:env5GConst.N_steps
    YError_plot_n1_ii = abs(YActual_plot_n1(:,ii) - YPred_plot_n1(:,ii));
    YError_plot_n2_ii = abs(YActual_plot_n2(:,ii) - YPred_plot_n2(:,ii));

    % YError_plot_nx structure: [mean; max; min]
    YError_plot_n1(:,ii) = [mean(YError_plot_n1_ii); max(YError_plot_n1_ii); min(YError_plot_n1_ii)];
    YError_plot_n2(:,ii) = [mean(YError_plot_n2_ii); max(YError_plot_n2_ii); min(YError_plot_n2_ii)];
end
%% Performance assessment

figure(1) 
% n1 (first considered response feature)
subplot(2,2,1);
plot(1:env5GConst.N_steps,(env5GConst.N_Dev/10).*[YActual_plot_n1_avg; YPred_plot_n1]); grid on;
ylim([0 2e3]);
legend('Actual','Predicted');
xlabel('SIB2 Slot'); ylabel('n_1');

% n2 (second considered response feature)
subplot(2,2,2);
plot(1:env5GConst.N_steps,(env5GConst.N_Dev/10).*[YActual_plot_n2_avg; YPred_plot_n2]); grid on;
ylim([0 2e3]);
legend('Actual','Predicted');
xlabel('SIB2 Slot'); ylabel('n_2');

subplot(2,2,3); 
plot(1:env5GConst.N_steps,YError_plot_n1(1,:),'r'); hold on;
plot(1:env5GConst.N_steps,YError_plot_n1(2:3,:),'--b'); hold off;
xlabel('SIB2 Slot'); ylabel('Absolute Error for n_1');
ylim([0 0.2]); grid on;

subplot(2,2,4); 
plot(1:env5GConst.N_steps,YError_plot_n2(1,:),'r'); hold on;
plot(1:env5GConst.N_steps,YError_plot_n2(2:3,:),'--b'); hold off;
xlabel('SIB2 Slot'); ylabel('Absolute Error for n_2');
ylim([0 0.2]); grid on;

figureName = strcat('Estimate_performance_type_',num2str(type),'_',num2str(no_layers),'_averaged.png');
saveas(1,figureName);

clear figureName;

%% Saving the file

save(FileName_fitrnet);