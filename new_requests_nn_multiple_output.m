%%  Deep Learning Network - Estimating New Requests
%% Getting Set Up
% Perform a workspace and figure reset

clear all; close all;
%% 
% Set the number of concantenated random episodes that make up the database:

no_it = 1e3; 

%% 
% Set the desired ACB policy, the type of normalization on the DNN's input
% training data, and the no. of DNN layers.

policy_type = 21; % P_ACB
SIB2 = 16; % No. of RAO channel within a SIB2 Slot
norm_type = 1; % Type of normalization
no_layers = 1; % Type of NN structure

%% 
% From the previous information generate the database's file name that will 
% be loaded and used to train and test the DNN.

% The obtained results will be stored in a file named 'FileName_fitrnet':
FileName_fitrnet = strcat('trained_net_',num2str(policy_type),...
    '_',num2str(SIB2),'_',num2str(norm_type),'_',num2str(no_layers));
% The loaded file containing the generated database is named
% 'FileName_database':
FileName_database = strcat('artificial_db_no_it_',num2str(no_it),'_',...
    num2str(policy_type),'_SIB2_',num2str(SIB2),'_Max_10');
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
        fullyConnectedLayer(4,"Name","fc_3")
        reluLayer("Name","relu_3")
        fullyConnectedLayer(2,"Name","fc_3")
        reluLayer("Name","relu_3")
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

% *** TEMPORARY MODIFICATION BEFORE BEING INCORPORATED IN THE DATASETS ***
% each sample must be the the average value over the no of RAO channels
% making up a SIB2 Slot.
data(:,1:4) = data(:,1:4)./SIB2;
data(:,6:7) = data(:,6:7)./SIB2;

% Cross validation (train: 70%, test: 30%)
cv = cvpartition(size(data,1),'HoldOut',0.3); % define a random partition on the dataset
idx = cv.test;

% Separate to training and test data
dataTrain = data(~idx,:);
dataValidation  = data(idx,:);

%% Normalizing training & validation data
% Normalization types:
%  *  0  =  No normalization
%  *  1  =  z-score
%  *  11 =  z-score with 1.5*sigma and an offset ensuring positive values,
%           P_ACB skipped
%  *  2  =  Power normalization
%  *  3  =  Arbitrary

if norm_type == 1
    [dataTrain,mu,sigma] = zscore(dataTrain); % by default std formula has N-1 in the denominator
    % Validation data is applied the same normalization as the training one
    dataValidation = (dataValidation - mu)./sigma;
elseif norm_type == 11
    % 4*sigma; re-scalated values to achieve positive values; P_ACB skipped
    [~,mu,sigma] = zscore(data(:,[1:4 6 7])); % by default std formula has N-1 in the denominator
    sigma = [sigma(1:4).*4 1 sigma(5:6).*4];
    mu = [mu(1:4) 0 mu(5:6)];
    dataTrain = (data - mu)./sigma; 
    dataTrain = dataTrain - [min(dataTrain(:,1:4),[],1) 0 min(dataTrain(:,6:7),[],1)];
    
    mu(1:4) = mu(1:4) - min(dataTrain(:,1:4),[],1);
    mu(6:7) = mu(6:7) - min(dataTrain(:,6:7),[],1);

    dataValidation = (dataValidation - mu)./sigma;
elseif norm_type == 2
    den_power = sqrt(mean(dataTrain.^2,1));
    dataTrain = dataTrain./den_power;

    % Validation data is applied the same normalization as the training one
    dataValidation = dataValidation./den_power;
elseif norm_type == 3
    dataTrain(:,[1 2 6 7]) = dataTrain(:,[1 2 6 7])./(env5GConst.N_Dev/10);
    dataTrain(:,3) = dataTrain(:,3)./(env5GConst.N_SIB2*env5GConst.M);    
    dataTrain(:,4) = dataTrain(:,4)./(env5GConst.N_SIB2*env5GConst.T_max_barr);

    % Validation data is applied the same normalization as the training one
    dataValidation(:,[1 2 6 7]) = dataValidation(:,[1 2 6 7])./(env5GConst.N_Dev/10);
    dataValidation(:,3) = dataValidation(:,3)./(env5GConst.N_SIB2*env5GConst.M);    
    dataValidation(:,4) = dataValidation(:,4)./(env5GConst.N_SIB2*env5GConst.T_max_barr);
end


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

%% Training the net
% Set training options

ValidationData_cellarray = {XValidation,YValidation};

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'ValidationData',ValidationData_cellarray, ...
    'ValidationFrequency',320/SIB2, ...
    'ValidationPatience',3200/SIB2, ...
    'MaxEpochs',30, ...
    'Verbose',1);
% The 'ValidationPatience' feature is used to stop training automatically 
% when the validation loss stops decreasing. In this case, if the loss on

% Loss = MSE when comparing predicted and actual response features
%% 
% Train the network:

format long;
[net, info] = trainNetwork(XTrain,YTrain,layers,options);

%%  Compute the validation RMSE:

YValidPred = predict(net,XValidation);

% de-normalize before computing error values:
if (norm_type == 1) || (norm_type == 11)
    YValidation = YValidation.*sigma(6:7)+mu(6:7);
    YValidPred = YValidPred.*sigma(6:7)+mu(6:7);
elseif norm_type == 3
    YValidation = YValidation.*(env5GConst.N_Dev/10);
    YValidPred = YValidPred.*(env5GConst.N_Dev/10);

end

format long;
validation_rmse = sum(sqrt(mean((YValidation - YValidPred).^2,1)))
%% Testing the net's performance on unseen data (test dataset)
% Load the test dataset:

no_it_test = 300;
% ****** TEMPORARILY ADDED *******
%policy_type = 0;

FileNameTest = strcat('artificial_db_no_it_',num2str(no_it_test),'_',num2str(policy_type),'_SIB2_',num2str(SIB2),'_TEST');
dataTest = load(FileNameTest,'data');
dataTest = dataTest.data;

dataTest = [dataTest(:,1:2) dataTest(:,4:5) dataTest(:,8) dataTest(:,6:7)];
dataTest(:,[1 2 3 4 6 7]) = dataTest(:,[1 2 3 4 6 7])./SIB2;

%%
% Normalize test data using the normalization constants obtained with
% training data:
if (norm_type == 1) || (norm_type == 11)
    dataTest = (dataTest - mu)./sigma;
elseif norm_type == 2
    dataTest = dataTest./den_power;
elseif norm_type == 3
    dataTest(:,[1 2 6 7]) = dataTest(:,[1 2 6 7])./(env5GConst.N_Dev/10);
    dataTest(:,3) = dataTest(:,3)./(env5GConst.N_SIB2*env5GConst.M);    
    dataTest(:,4) = dataTest(:,4)./(env5GConst.N_SIB2*env5GConst.T_max_barr);
end

XTest = dataTest(:,1:5); % get the first five features as input data
YTest = dataTest(:,6:7); % set 'n1' and 'n2' as responses
%% 
% Test the network's performance by measuring prediction accuracy:

YTestPred = predict(net,XTest);
%% 
% Evaluate the performance of the model by calculating the root-mean-square 
% error (RMSE) of the predicted and actual number of newly generated access requests:

% de-normalize before computing error values:
if (norm_type == 1) || (norm_type == 11)
    YTest = YTest.*sigma(6:7)+mu(6:7);
    YTestPred = YTestPred.*sigma(6:7)+mu(6:7);
elseif norm_type == 3
    YTest = YTest.*(env5GConst.N_Dev/10);
    YTestPred = YTestPred.*(env5GConst.N_Dev/10);

end


format long;
test_rmse_n1 = sqrt(mean((YTest(:,1) - YTestPred(:,1)).^2,1))
test_rmse_n2 = sqrt(mean((YTest(:,2) - YTestPred(:,2)).^2,1))

test_rmse = sum(sqrt(mean((YTest - YTestPred).^2,1)))

%% Performance Assessment

YActual_plot_n1 = zeros(no_it_test,env5GConst.N_steps);
YActual_plot_n2 = zeros(no_it_test,env5GConst.N_steps);

YPred_plot_n1 = zeros(no_it_test,env5GConst.N_steps);
YPred_plot_n2 = zeros(no_it_test,env5GConst.N_steps);

YError_plot_n1 = zeros(3,env5GConst.N_steps);
YError_plot_n2 = zeros(3,env5GConst.N_steps);

for it_part = 1:1:no_it_test
    %it_part = 200; % set iteration partition to visualize performance of the trained model
    dataActual_plot = dataTest(1+(it_part-1)*env5GConst.N_steps:it_part*env5GConst.N_steps,:);

    YPred_plot = predict(net,dataActual_plot(:,1:5));

    if (norm_type == 1) || (norm_type == 11)

        % De-normalize the response data to compute prediction error
        YActual_plot_n1(it_part,:) = dataActual_plot(:,6)'.*sigma(6)+mu(6);
        YActual_plot_n2(it_part,:) = dataActual_plot(:,7)'.*sigma(7)+mu(7);
    
        % De-normalize predicted responses to compute prediction error
        YPred_plot(:,1) = YPred_plot(:,1).*sigma(6)+mu(6); % de-mormalize n_1
        YPred_plot(:,2) = YPred_plot(:,2).*sigma(7)+mu(7); % de-mormalize n_2

    % NOTE: normalization type #2 not implemented as it has been dismissed
    % eventually

    elseif norm_type == 3

        YActual_plot_n1(it_part,:) = dataActual_plot(:,6)'.*(env5GConst.N_Dev/10);
        YActual_plot_n2(it_part,:) = dataActual_plot(:,7)'.*(env5GConst.N_Dev/10);

        % De-normalize predicted responses to compute prediction error
        YPred_plot = YPred_plot.*(env5GConst.N_Dev/10); 

    end

    YPred_plot_n1(it_part,:) = YPred_plot(:,1)'; YPred_plot_n2(it_part,:) = YPred_plot(:,2)';
end

YActual_plot_n1_avg = mean(YActual_plot_n1,1); YActual_plot_n2_avg = mean(YActual_plot_n2,1);
YPred_plot_n1 = mean(YPred_plot_n1,1); YPred_plot_n2 = mean(YPred_plot_n2,1);


for ii = 1:env5GConst.N_steps
    % Computing the relative error
    YError_plot_n1_ii = abs(YActual_plot_n1(:,ii) - YPred_plot_n1(:,ii));
    YError_plot_n2_ii = abs(YActual_plot_n2(:,ii) - YPred_plot_n2(:,ii));

    % YError_plot_nx structure: [mean; max; min]
    YError_plot_n1(:,ii) = [mean(YError_plot_n1_ii); max(YError_plot_n1_ii); min(YError_plot_n1_ii)];
    YError_plot_n2(:,ii) = [mean(YError_plot_n2_ii); max(YError_plot_n2_ii); min(YError_plot_n2_ii)];
end

%% Plots - Graphic Performance Assessment

figure(1) 
% n1 (first considered response feature)
subplot(1,2,1);
plot(1:env5GConst.N_steps,[YActual_plot_n1_avg; YPred_plot_n1]); grid on;
legend('Actual','Predicted');
xlabel('SIB2 Slot'); ylabel('n_1');

% n2 (second considered response feature)
subplot(1,2,2);
plot(1:env5GConst.N_steps,[YActual_plot_n2_avg; YPred_plot_n2]); grid on;
legend('Actual','Predicted'); 
xlabel('SIB2 Slot'); ylabel('n_2');

figureName = strcat('Episode_performance_',num2str(policy_type),'_',num2str(SIB2),'_',num2str(norm_type),'_',num2str(no_layers),'_averaged.png');
saveas(1,figureName);

figure(2)
subplot(1,2,1); 
plot(1:env5GConst.N_steps,YError_plot_n1(1,:),'r','DisplayName','Mean'); hold on;
plot(1:env5GConst.N_steps,YError_plot_n1(2:3,:),'--b',...
    'DisplayName','Max. / Min.'); hold off;
xlabel('SIB2 Slot'); ylabel('Absolute Error for n_1'); 
legend('location','best');
grid on;

subplot(1,2,2); 
plot(1:env5GConst.N_steps,YError_plot_n2(1,:),'r','DisplayName','Mean'); hold on;
plot(1:env5GConst.N_steps,YError_plot_n2(2:3,:),'--b',...
    'DisplayName','Max. / Min.'); hold off;
xlabel('SIB2 Slot'); ylabel('Absolute Error for n_2');
legend('location','best')
grid on;

figureName = strcat('Episode_Absolute_Error_',num2str(policy_type),'_',num2str(SIB2),'_',num2str(norm_type),'_',num2str(no_layers),'_averaged.png');
saveas(2,figureName);

clear figureName;

%% Saving the file
save(FileName_fitrnet);