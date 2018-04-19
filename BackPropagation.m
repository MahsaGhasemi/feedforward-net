%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dJ] = BackPropagation(m,n,p,act_func,X,Y,V,Y_hat,Z,d_sigma)
% This function finds the gradient of the cost function
% Inputs:
%   m -> number of inputs
%   n -> number of nodes in hidden layer
%   p -> number of outputs
%   act_func -> determines activation function
%       if '1': sigmoid
%       if '2': tangent hyperbolic
%       if '3': radial basis function
%   X -> the instance 'k' input
%   Y -> the instance 'k' output
%   V -> the current concatenated weight
%   Y_hat -> the instance 'k' estimated output
%   Z -> output of hidden layer
%   d_sigma -> derivative of activation functions
% Outputs:
%   dJ -> gradient of cost function

if act_func==1 % sigmoid
    % decoding weights and biases of each layer
    W2 = reshape(V(1:n*m), [n,m]);
    B2 = reshape(V(n*m+1:n*m+n), [n,1]);
    W3 = reshape(V(n*m+n+1:n*m+n+p*n), [p,n]);
    B3 = reshape(V(n*m+n+p*n+1:n*m+n+p*n+p), [p,1]);

elseif act_func==2 % tangent hyperbolic
    % decoding weights and biases of each layer
    W2 = reshape(V(1:n*m), [n,m]);
    B2 = reshape(V(n*m+1:n*m+n), [n,1]);
    W3 = reshape(V(n*m+n+1:n*m+n+p*n), [p,n]);
    B3 = reshape(V(n*m+n+p*n+1:n*m+n+p*n+p), [p,1]);
    
elseif act_func==3 % radial basis function
    % decoding weights and biases of each layer
    W3 = reshape(V(1:p*n), [p,n]);
    B3 = reshape(V(p*n+1:p*n+p), [p,1]);
end

% computing the derivatives
if size(d_sigma,1) > 0
    dy_dB3 = eye(p);
    dy_dW3 = zeros(p,n*p);
    for i=0 : n-1
        dy_dW3(1:p,i*p+1:(i+1)*p) = Z(i+1)*eye(p);
    end
    dy_dB2 = W3 * diag(d_sigma) * eye(n);
    X_c = zeros(n,m*n);
    for i=0 : m-1
        X_c(1:n,i*n+1:(i+1)*n) = X(i+1)*eye(n);
    end
    dy_dW2 = W3 * diag(d_sigma) * X_c;
    
    de = -[dy_dW2, dy_dB2, dy_dW3, dy_dB3];

else % the case that radial basis function is chosen
    if m==1
        n_new = n;
    elseif m==2
        n_new = (ceil(sqrt(n)))^2;
    end
    dy_dB3 = eye(p);
    dy_dW3 = zeros(p,n_new*p);
    for i=0 : n_new-1
        dy_dW3(1:p,i*p+1:(i+1)*p) = Z(i+1)*eye(p);
    end
    de = -[dy_dW3, dy_dB3];
end

e = Y-Y_hat; % output error
dJ = 2*e'*de; % gradient caused by this data point

end