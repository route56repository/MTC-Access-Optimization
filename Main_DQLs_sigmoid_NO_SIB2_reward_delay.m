%% Initialization

clear all; close all;
%% User-defined inputs

%MaxEpi = input("Enter maximum number of episodes: "); 
% Command asks for the max. no of episodes when executing the script

MaxEpi = 1e4; actifun = 'sigmoid';

%FileName=strcat('DQLs_1c9_',num2str(MaxEpi));
FileName = strcat('DQLs_results_',date,'_',actifun,'_',num2str(MaxEpi),'_reward_delay');
%% Create an observation specification object
% Define the observation space as a continuous four-dimensional space, so that 
% a single observation is a column vector containing four doubles.

obsInfo = rlNumericSpec([4 1]); % 4-by-1 dimensional space
%% 
% Several property values assigned to the object 'obsInfo':

obsInfo.Name = 'observations'; % property value 'name' assigned to rlNumericSpec object
obsInfo.Description = 'Served,Collided,Delay,P_ACB';
obsInfo.LowerLimit = [0 0 0 0]; % NOTE: when LowerLimit is specified as a scalar rlNumericSpec 
% applies it to all entries in the data space.
obsInfo.UpperLimit = [2 3 2 1]; 
%% Create an action specification object
% The action space is defined as a discrete (limited number of actions only) 
% one-dimensional object

elements=0.05:0.05:1;
actInfo = rlFiniteSetSpec(elements);% Specify valid numeric values for a
        % single action or single observation.
actInfo.Name = 'actions';
actInfo.Description = 'Quantified P_ACB';
clear elements
%% Create my 5G environment
% Create the structure that contains the environment constants.

env5GConst.T=10;  % Episode or Frame duration in sec.
env5GConst.T_RAO=0.005;  % RAO time in sec.
env5GConst.N_RAO = env5GConst.T/env5GConst.T_RAO; % Number of RAO channels in a frame
env5GConst.MaxRAO = ceil(1.5*env5GConst.N_RAO);  % Number of actual RAO channels during which the system's evolution is observed
env5GConst.N_SIB2 = 1;  % no of RAO channels during which state stats (including P_ACB) are not updated
env5GConst.N_steps = round(env5GConst.MaxRAO/env5GConst.N_SIB2);  % no of times that state stats are updated
env5GConst.MaxWait = 10;  % max no of device collisions
env5GConst.Ac_Pacb=0.05:0.05:1; % Access Class Barring Factor
env5GConst.M = 54;  % Number of available simultaneous communications
env5GConst.N_Dev = 3e4;  % Number of MTC devices in scheduled traffic model
env5GConst.AgentType = 1;  % 1 DQL singleOut 2 DQL MultipleOut 3 DDPG Action Numeric
env5GConst.Norm = 1;  % Normalize Input, recommended with sigmoid

%% 
% Create an anonymous function handle to the custom step function, passing env5GConst 
% as an additional input argument.
% 
% Because env5GConst is available at the time that StepHandle is created, the 
% function handle includes those values.
% 
% The values persist within the function handle even if you clear the variables.

Step5GHandle = @(Action,LoggedSignals) my5GStepFunction_SIB2_reward_delay(Action,LoggedSignals,env5GConst);
% Use the same reset function, specifying it as a function handle rather than by using its name.
Reset5GHandle = @()my5GResetFunction_SIB2(env5GConst);
%% 
% Create the environment using the custom function handles.

env5G = rlFunctionEnv(obsInfo,actInfo,Step5GHandle,Reset5GHandle);
% env5G = rlFunctionEnv(obsInfo,actInfo,'my5GStepFunction','my5GResetFunction');
%% Create a deep neural network to approximate the Q-value function
% The network must have two inputs, one for the observation and one for the 
% action. The observation input (here called myobs) must accept a four-element 
% vector (the observation vector defined by obsInfo). The action input (here called 
% myact) must accept a one-element vector (the action vector defined by actInfo). 
% The output of the network must be a scalar, representing the expected cumulative 
% long-term reward when the agent starts from the given observation and takes 
% the given action.
% 
% This model has been inspired in 
% 
% <https://es.mathworks.com/help/reinforcement-learning/ref/rldqnagent.html 
% https://es.mathworks.com/help/reinforcement-learning/ref/rldqnagent.html>
% 
% see a diferent option in this example
% 
% <https://es.mathworks.com/help/reinforcement-learning/ug/train-dqn-agent-to-balance-cart-pole-system.html 
% https://es.mathworks.com/help/reinforcement-learning/ug/train-dqn-agent-to-balance-cart-pole-system.html>
% 
% Na outputs where Na is the number of actions

% create a DEEP NEURAL NETWORK approximator 
% the observation input layer must have 4 elements (obsInfo.Dimension(1))
% the action input layer must have 1 element (actInfo.Dimension(1))
% the output must be a scalar
statePath = featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'myObs');
actionPath = featureInputLayer(actInfo.Dimension(1), 'Normalization', 'none', 'Name', 'myAct');
commonPath = [concatenationLayer(1,2,'Name','concat') % indicating one single ouput?
    fullyConnectedLayer(10, 'Name', 'CriticComFC1') % basicament es crear una 'layer' intermitja de forma habitual (?)
    % all inputs are weighted and biased to generate a 10-node output
    % scheme
    %reluLayer('Name','CriticCommonRelu')
    %sigmoidLayer('Name', 'CriticComsig1') % apply sigmoid regularization at each output
    %tanhLayer("Name","tanh_activation") % TANH unction as an inter-layer activation function
%     fullyConnectedLayer(10, 'Name', 'CriticComFC2')
%   reluLayer('Name','CriticCommonRelu')
    sigmoidLayer('Name', 'CriticComsig2')
    fullyConnectedLayer(1, 'Name', 'prev_output')]; % new inputs weighted and summed up to produce a one-node output (Q function)
    %sigmoidLayer('Name','output')
    %regressionLayer('Name','output_regression')]; % output regression layer added
    % additionLayer(2,'Name', 'add') 
   % concatenationLayer(5,2,'Name','concat');
net = layerGraph(statePath);
net = addLayers(net, actionPath);
net = addLayers(net, commonPath);    
net = connectLayers(net,'myObs','concat/in1'); % to connect to the 'concat' node, it must indicated like this (see documentation)
net = connectLayers(net,'myAct','concat/in2');
%plot(net)
%analyzeNetwork(net)
%% Create the critic with rlQValueRepresentation
% Create Q-Value Function Critic from Deep Neural Network
% 
% <https://es.mathworks.com/help/reinforcement-learning/ref/rldqnagent.html#mw_4b8dc3d0-2667-44d5-ba22-aeae022bda64 
% https://es.mathworks.com/help/reinforcement-learning/ref/rldqnagent.html#mw_4b8dc3d0-2667-44d5-ba22-aeae022bda64>

%criticOpts = rlRepresentationOptions('LearnRate',0.001,'GradientThreshold',10);
criticOpts = rlRepresentationOptions('LearnRate',0.001);
% This object implements a Q-value function approximator to be used as a critic within a reinforcement learning agent
critic = rlQValueRepresentation(net,obsInfo,actInfo, ...
    'Observation',{'myObs'},'Action',{'myAct'},criticOpts);
% s=[0 0 0 1]; % Initial State
% a=1; % Action
% Q = getValue(critic,s,a); % Check the critic using the current network weights
%% Create Q-Learning Agent
% Create a Q-Learning Agent Options Object

agentOpt = rlDQNAgentOptions;% Specify some training options for the critic representation using rlRepresentationOptions.
%initOpts = rlAgentInitializationOptions('NumHiddenUnit',128);
agentOpt.DiscountFactor = 0.9; % the agent considers most past actions
agentOpt.TargetSmoothFactor=0.001;
agentOpt.EpsilonGreedyExploration.Epsilon = 0.9;
agentOpt.EpsilonGreedyExploration.EpsilonDecay = 0.000001; % MIRAR APUNTS REU !!!!!!!!
% Epsilon is updated using the following formula when it is greater than EpsilonMin:
% Epsilon = Epsilon*(1-EpsilonDecay)
% exploring stage is discouraged as more and more knowledge is accumulated
agentOpt.EpsilonGreedyExploration.EpsilonMin=0.01;
% agentOpt.SaveExperienceBufferWithAgent=1;USEFUL IF A SECOND TRAINING CALL
% IS PERFORMED, IN ORDER TO USE THE EXPERIENCEBUFFER TO START TRAINING
% critic based on a mini-batch of experiences randomly sampled from the buffer.
agentOpt.MiniBatchSize = 32; % Sample a random mini-batch 
agentOpt.ExperienceBufferLength=2000; % The agent updates the
% of M experiences (Si,Ai,Ri,S'i) from the experience buffer to update loss
% function
agent5G = rlDQNAgent(critic,agentOpt); % MIRAR DOCUMENTATTION: diferencia amb fer servir obsInfo, actInfo?? !!!!
% Acc=getAction(agent5G,{rand(obsInfo(1).Dimension)}); % Check the agent generating a random action
inicriticParams = getLearnableParameters(getCritic(agent5G)); % Check the initial weights of the critic net
%% Train Reinforcement Learning Agents
% Once you have created an environment and reinforcement learning agent, you 
% can train the agent in the environment using the train function. For example, 
% create a training option set opt, and train agent agent5G in environment env5G.

opt = rlTrainingOptions(...
    'MaxEpisodes',MaxEpi,...
    'MaxStepsPerEpisode',env5GConst.N_steps,...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',0,... % stop only if there is an agent that provides zero avg delay
    'UseParallel',0, ...
    'SaveAgentCriteria',"EpisodeReward",... 
    'SaveAgentValue',-7e5,... % save all agent achieving a reward = -avg delay greater than the stated value
    'SaveAgentDirectory', strcat(pwd,'\Agents'),...
    'Verbose',1,...  % print the iteration number and result
    'Plots','none'); % Activate/Deactivate RL Window
tic
trainStats = train(agent5G,env5G,opt);
toc
save(FileName)