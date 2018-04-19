%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [V,J_history,J_his_test] = Learning_VGD(m,n,p,data,data_test,act_func)
% Learning the network using gradient descent with adaptive learning rate
% This function finds the ANN weights
% Inputs:
%   m -> number of inputs
%   n -> number of nodes in hidden layer
%   p -> number of outputs
%   data -> training data
%   data_test -> testing data
%   act_func -> activation function selection
% Outputs:
%   V -> weights of NN
%   J_history -> record of cost function
%   J_his_test -> record of cost function for testing set

n_tr = size(data,2); % number of data points
X_range = [min(data(1:m,:),[],2), max(data(1:m,:),[],2)]; %input range

%% initializing weights and biases

if act_func==1 % sigmoid
    W2 = 2*rand(n,m)-ones(n,m);
    B2 = 2*rand(n,1)-ones(n,1);
    W3 = 2*rand(p,n)-ones(p,n);
    B3 = 2*rand(p,1)-ones(p,1);
    
    % concatenating weights and biases into V
    V = [W2(:); B2; W3(:); B3];
    
elseif act_func==2 % tangent hyperbolic
    W2 = 2*rand(n,m)-ones(n,m);
    B2 = 2*rand(n,1)-ones(n,1);
    W3 = 2*rand(p,n)-ones(p,n);
    B3 = 2*rand(p,1)-ones(p,1);
    
    % concatenating weights and biases into V
    V = [W2(:); B2; W3(:); B3];
    
elseif act_func==3 % radial basis function
    W3 = 2*rand(p,n)-ones(p,n);
    B3 = 2*rand(p,1)-ones(p,1);
    
    % concatenating weights and biases into V
    V = [W3(:); B3];
end

%% training

epoch = 0;
epoch_max = 500; % maximum number of iterations
J_min = 0.01; % target minimum for cost function
converge = 0; % 1 if converged and 0 if not
J_history = zeros(1,epoch_max);
J_his_test = zeros(1,epoch_max);

test_win = 40; % window size for testing set
J_win = NaN(1,test_win);
dJ_win = -1; % change in cost function of a data window 'test_win'

eta = 0.05; % step size
mo = 0.5; % momentum
dV_pre = zeros(size(V)); % delta(weight) of previous epoch

l_inc = 1.05;
l_dec = 0.7;
max_p_inc = 1.05;

h_bar = waitbar(0,'Iniatializing','Name','Computing ANN weights ...',...
    'CreateCancelBtn',...
    'setappdata(gcbf,''canceling'',1)');
setappdata(h_bar,'canceling',0)

% opening the file for writing the cost functions
fopen('learning.dat','w');
fid = fopen('learning.dat','a+');
fprintf(fid,'%12s %12s %12s\r\n','iteration','J_train','J_test');

while ~converge
    J = 0; % cost function
    dJ = zeros(size(V')); % gradient of cost function
    
    for i=1 : n_tr
        X = data(1:m,i); % reading input 'k'
        Y = data(m+1:m+p,i); % reading output 'k'
        [Z,Y_hat,d_sigma] = FeedForward(m,n,p,act_func,X,X_range,V);
        dJ = dJ + BackPropagation(m,n,p,act_func,X,Y,V,Y_hat,Z,d_sigma);
        e = Y-Y_hat;
        J = J + 1/n_tr*(e'*e); % cumulating cost function
    end
    
    dV = mo*dV_pre - eta*mo*dJ';
    J_new = 0;
    
    for i=1 : n_tr
        X = data(1:m,i); % reading input 'k'
        Y = data(m+1:m+p,i); % reading output 'k'
        [~,Y_hat,~] = FeedForward(m,n,p,act_func,X,X_range,V+dV);
        e = Y-Y_hat;
        J_new = J_new + 1/n_tr*(e'*e); % cumulating cost function
    end
    
    epoch = epoch + 1;
    
    % modifiying the learning rate accordingly
    if J_new < J % increasing the learning rate
        eta = l_inc*eta;
        V = V + dV;
        dV_pre = dV;
    elseif J_new/J > max_p_inc % decreasing the learning rate
        eta = l_dec*eta;
    else % keeping the learning rate
        V = V + dV;
        dV_pre = dV;
    end
    
    J_history(epoch) = J;
    
    % cost function record over a window
    J_win = circshift(J_win,[0,-1]);
    J_win(test_win) = Testing(m,n,p,data_test,V,act_func);
    dJ_win = mean(diff(J_win));
    
    J_his_test(epoch) = J_win(test_win);
    fprintf(fid,'%12f %12.6f %12.6f\r\n',...
        [epoch, J_history(epoch), J_his_test(epoch)]); % writing cost functions to file
    
    if getappdata(h_bar,'canceling')
        break
    end
    waitbar(min(max([epoch/epoch_max,J_min/J]),1),h_bar,...
        ['Number of iterations: ',sprintf('%6.0f',epoch)]);
    
    if (epoch > epoch_max) || (J < J_min) %|| (dJ_win > 0.01) % termination condition
        converge = 1;
    end
end

fclose(fid);
delete(h_bar);

end