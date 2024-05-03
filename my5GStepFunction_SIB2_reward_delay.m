function [NextObs,Reward,IsDone,LogSig] = my5GStepFunction_SIB2_reward_delay(Action,LogSig,env5GConst)
% Custom step function to construct the 5G environment for the function
% name case.
%
% ACTION IS CATEGORIC
% STATE IS NORMALIZED IF env5GConst.Norm=1
%
% This function applies the given action to the environment and evaluates
% the system dynamics for one simulation step.

%% Check if the given action is valid.
if ~ismember(Action,env5GConst.Ac_Pacb)
    error('Action is not a valid action');
else
    P_ACB=Action;
end
IsDone = 0;

%% Unpack the state vector from the logged signals.
% s = LoggedSignals.State;

% Resetting the counter before beginning the outer loop (state and action
% update frame)
N_served = 0; N_delayed = 0; Avg_delay = 0;

% ** NEW **: Actual no of collisions (not collided devices)
N_collisions = 0;

for ii = 1:env5GConst.N_SIB2

    %% Recording access requests
    Index_request=LogSig.MTC_feat(LogSig.MTC_RAOslot == (env5GConst.N_SIB2*(LogSig.n-1))+ii,1); % Index of active MTC devices in n^th RAO
    % REMINDER: MTC_RAOslot = RAO channel each device is allocated to

    %% Access Class Barring
    % Accessing media by using ACB factor
    q_var=rand(length(Index_request),1);
    %Index_delayed=Index_request((q_var > P_ACB) && (MTC_feat(Index_request,3) == 0));

    % Delayed/barred devices (those that do not pass the ACB check) are recorded
    % NOTE: previously barred devices are transferred directly to the preamble
    % designation stage, that is previously delayed devices skip the ACB check
    Index_delayed = []; Index_request_new = Index_request;
    for jj = 1:length(Index_request)
        % if a device does not pass the ACK check and has not been previously delayed, ...
        if ((q_var(jj) > P_ACB) && (LogSig.MTC_feat(Index_request(jj),3) == 0)) 
            % ... this device is delayed
            Index_delayed = [Index_delayed; Index_request(jj)];   
            Index_request_new(jj) = 0; % delayed MTCDs are removed from this slot
        % if a device exceeds the max allowed no of barrings + collisions, ...
        elseif LogSig.MTC_feat(Index_request(jj),3) + LogSig.MTC_feat(Index_request(jj),4) > env5GConst.MaxWait + 1
            % ... the device is not allowed to access the system again
            Index_request_new(jj) = 0; % MTCDs that collisioned over env5GConst.MaxWait times are removed from this slot
            % NOTE: As I understand from the non-deep QL version, dismissing
            % over-collisioned MTCDs is performed right after passing the ACB
            % check.
        end
    end
    Index_request(Index_request_new == 0) = []; % remove delayed MTCDs 
    clear ii Index_request_new

    T_barring=(0.7*ones(length(Index_delayed),1)+0.6*rand(length(Index_delayed),1))...
        .*(2.^LogSig.MTC_feat(Index_delayed,3)); % Tbarring in sec.
    T_barring=ceil(T_barring/env5GConst.T_RAO); % Tbarring in normalized time
    LogSig.MTC_feat(Index_delayed,2)=LogSig.MTC_feat(Index_delayed,2)+T_barring; % Delay
    LogSig.MTC_RAOslot(Index_delayed)=LogSig.MTC_RAOslot(Index_delayed)+T_barring';
    LogSig.MTC_feat(Index_delayed,3)=LogSig.MTC_feat(Index_delayed,3)+1; % Number of delays

    %% Preamble generation and designation     
    Preamble=randi(env5GConst.M,[length(Index_request),1]); % M = no of available simultaneous communications
    [Preamble,Idx]=sort(Preamble);
    Index_request=Index_request(Idx);

    Prea_repe = zeros(size(Preamble,1)+1,1);
    Prea_expand = [0; Preamble];
    for i1 = 2:length(Prea_expand)-1
        if Prea_expand(i1) == Prea_expand(i1+1)
            Prea_repe(i1:i1+1) = 1;
            if Prea_expand(i1) ~= Prea_expand(i1-1)
                N_collisions = N_collisions + 1;
            end
        end
    end
    Prea_repe = Prea_repe(2:end); % remove the first element (additional element resulting form the expanded preamble version)
    
    Index_collided=Index_request(Prea_repe==1);
    T_BO=rand(length(Index_collided),1).*(2.^LogSig.MTC_feat(Index_collided,3)); % TBO in sec.
    T_BO=ceil(T_BO/env5GConst.T_RAO); % TBO in normaliced time
    LogSig.MTC_feat(Index_collided,2)=LogSig.MTC_feat(Index_collided,2)+T_BO; % Delay
    LogSig.MTC_feat(Index_collided,4)=LogSig.MTC_feat(Index_collided,4)+1; % Number of collisions
    LogSig.MTC_RAOslot(Index_collided)=LogSig.MTC_RAOslot(Index_collided)+T_BO';

    % Succesfull Transmisions
    Index_request=Index_request(Prea_repe==0);
    LogSig.MTC_feat(Index_request,5)=1; % Succesfull transmisions
    
    N_served = N_served + length(Index_request);
    N_delayed = N_delayed + length(Index_delayed);
    %N_collided = N_collided + length(Index_collided);
    if ~isempty(Index_request)
        Avg_delay = Avg_delay + mean(LogSig.MTC_feat(Index_request,2));
    end
end

% Filling State Matrix
LogSig.St_Mat(LogSig.n,1) = N_served;
LogSig.St_Mat(LogSig.n,2) = N_delayed;
LogSig.St_Mat(LogSig.n,3) = N_collisions;
LogSig.St_Mat(LogSig.n,4) = Avg_delay;
LogSig.St_Mat(LogSig.n,5) = P_ACB; % Access Class Barring Factor

% New State
sp = [N_served; N_collisions; Avg_delay; P_ACB];

if env5GConst.Norm == 1 % Normalize Input
    sp = [sp(1)*2/env5GConst.M; sp(2)/env5GConst.M; sp(3)*2/env5GConst.MaxRAO; sp(4)];
end

%% Return updated matrices
LogSig.n=LogSig.n+1;
LogSig.State = sp;
NextObs=sp;

% Check terminal condition.
if LogSig.n==env5GConst.MaxRAO
    IsDone = 1;
end

% Get reward.
%Reward = N_served;
Reward = -Avg_delay;
end