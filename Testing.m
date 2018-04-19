%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [J_test] = Testing(m,n,p,data,V,act_func)
% Test the ANN
% This function tests the ANN
% Inputs:
%   m -> number of inputs
%   n -> number of nodes in hidden layer
%   p -> number of outputs
%   data -> testing data
%   V -> weights and biases
%   act_func -> activation function selection
% Outputs:
%   J_test -> cost function for the test data

Np = size(data,2); % number of data points
X_range = [min(data(1:m,:),[],2), max(data(1:m,:),[],2)];
J_test = 0; % cost function

for i=1 : Np
    X = data(1:m,i); % reading input 'k'
    Y = data(m+1:m+p,i); % reading output 'k'
    Y_hat = NeuralNet(m,n,p,act_func,X,X_range,V);
    e = Y-Y_hat;
    J_test = J_test + 1/Np*(e'*e); % cumulating cost function
end
    
end