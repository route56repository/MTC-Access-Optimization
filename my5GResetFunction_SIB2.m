function [InitialObservation, LoggedSignals] = my5GResetFunction_SIB2(env5GConst)
% Reset function as required by MATLAB's RL environment. Reset all data
% containers and other functiones to initial state.
% ***************************************************************
%    Input:
% * env5GConst   = contains all the constants that are needed to generate
%                  the initial-state conditions
%
% *******************************************      ********************
%    Output:
%
% * InitialObservation = save the initial state stat
% * LoggedSignals      = object containing the matrices MTC_feat,
%                        MTC_RAOslot and St_Mat
% ***************************************************************
%% Generate random access times for participating devices
Beta_object = makedist('beta','a',3,'b',4);
Beta_vars = random(Beta_object,[1,env5GConst.N_Dev])*env5GConst.T;
MTC_RAOslot = ceil(Beta_vars/env5GConst.T_RAO); 

%% Generate data containers
%MTC_feat     = N_Dev x 5 matrix containing the system's current stats. 
%               Specifically, the columns are distributed as follows:
%
%                  1) MTC Device number
%                  2) Delay in number of RAO channels
%                  3) number of failed ACB checks(Nbarring)
%                  4) number of collisions
%                  5) 0 / 1(served)

MTC_feat = zeros(env5GConst.N_Dev,5); 
MTC_feat(:,1) = 1:env5GConst.N_Dev;%MTC devices are ordered from 1 to N_dev

% St_Mat        = N_RAO x 5 matrix containing state space stats. The
%                 columns are distributed as follows:
%
%                  1) No of served devices
%                  2) No of delayed devices
%                  3) No of collided devices
%                  4) Avg Delay
%                  5) P_ACB (action)

St_Mat = zeros(env5GConst.N_steps,5); 

%% Saving data containers as LoggedSignals
LoggedSignals.MTC_feat=MTC_feat;
LoggedSignals.MTC_RAOslot=MTC_RAOslot;
LoggedSignals.St_Mat=St_Mat;
LoggedSignals.n=1;

%% Setting initial state conditions 
LoggedSignals.State = [0;0;0;1];
InitialObservation = LoggedSignals.State;
end