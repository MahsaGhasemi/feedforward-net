%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Z,Y_hat,d_sigma] = FeedForward(m,n,p,act_func,X,X_range,V)
% FeedForward
% This function uses the current weights to do feedforward
% Inputs:
%   m -> number of inputs
%   n -> number of nodes in hidden layer
%   p -> number of outputs
%   act_func -> determines activation function
%       if '1': sigmoid
%       if '2': tangent hyperbolic
%       if '3': radial basis function
%   X -> the instance 'k' input
%   X_range -> range of data
%   V -> the current concatenated weight
% Outputs:
%   Z -> output of hidden layer
%   d_sigma -> derivative of activation functions

if act_func==1 % sigmoid
    % decoding weights and biases of each layer
    W2 = reshape(V(1:n*m), [n,m]);
    B2 = reshape(V(n*m+1:n*m+n), [n,1]);
    W3 = reshape(V(n*m+n+1:n*m+n+p*n), [p,n]);
    B3 = reshape(V(n*m+n+p*n+1:n*m+n+p*n+p), [p,1]);
    Xi = W2*X + B2;
    
    % computing output
    Z = logsig(Xi);
    d_sigma = Z.*(1-Z);
    Y_hat = W3*Z + B3;
    
elseif act_func==2 % tangent hyperbolic
    % decoding weights and biases of each layer
    W2 = reshape(V(1:n*m), [n,m]);
    B2 = reshape(V(n*m+1:n*m+n), [n,1]);
    W3 = reshape(V(n*m+n+1:n*m+n+p*n), [p,n]);
    B3 = reshape(V(n*m+n+p*n+1:n*m+n+p*n+p), [p,1]);
    Xi = W2*X + B2;
    
    % computing output
    Z = tanh(Xi);
    d_sigma = 1-Z.^2;
    Y_hat = W3*Z + B3;
    
elseif act_func==3 % radial basis function
    % decoding weights and biases of each layer
    W3 = reshape(V(1:p*n), [p,n]);
    B3 = reshape(V(p*n+1:p*n+p), [p,1]);
    
    if n==1
        if m==1 % 1D Gaussian
            c = (X_ramge(2)-X_range(1))/2; % expected value
            w = 2*c; % variance
            Z = exp(-(X-c)' * (w^(-2)) * (X-c)); % calculating Gaussian function for given input
            d_sigma = []; % dummy variable here
        elseif m==2 % 2D Gaussian
            c = [(X_ramge(1,2)-X_range(1,1))/2; (X_ramge(2,2)-X_range(2,1))/2]; % expected value
            w = 2*c; % variance
            Z = exp(-(X-c)' * diag(w.^(-2)) * (X-c)); % calculating Gaussian function for given input
            d_sigma = []; % dummy variable here
        end
        
    else
        if m==1 % 1D Gaussian
            w = (X_range(2)-X_range(1))/(n-1); % variance
            c = linspace(X_range(1), X_range(2), n); % expected values
            Z = zeros(n,1);
            
            for i=1 : n % calculating Gaussian functions for given input
                Z(i) = exp(-(X-c(i))' * (w^(-2)) * (X-c(i)));
            end
            d_sigma = []; % dummy variable here
            
        elseif m==2 % 2D Gaussian
            n_new = ceil(sqrt(n));
            w = (X_range(:,2)-X_range(:,1))/(n_new-1); % variance
            [cx,cy] = meshgrid(linspace(X_range(1,1), X_range(1,2), n_new),...
                linspace(X_range(2,1), X_range(2,2), n_new)); % expected values
            c = [cx(:)'; cy(:)']; % concatenating expected values
            n_new = n_new^2;
            Z = zeros(n_new,1);
            
            for i=1 : n_new % calculating Gaussian functions for given input
                Z(i) = exp(-(X-c(:,i))' * diag(w.^(-2)) * (X-c(:,i)));
            end
            d_sigma = []; % dummy variable here
        end
    end
        
    Y_hat = W3*Z + B3;
end

end