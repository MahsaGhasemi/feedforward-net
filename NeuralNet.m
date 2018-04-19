%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Y_hat] = NeuralNet(m,n,p,act_func,X,X_range,V)
% NeuralNet
% This function approximates the function based on the given NN
% Inputs:
%   m -> number of inputs
%   n -> number of nodes in hidden layer
%   p -> number of outputs
%   act_func -> determines activation function
%       if '1': sigmoid
%       if '2': tangent hyperbolic
%       if '3': radial basis function
%   X -> the instance 'k'
%   X_range -> range of data
%   V -> the current concatenated weight
% Outputs:
%   Y_hat -> approximated output

if act_func==1 % sigmoid
    % decoding weights and biases of each layer
    W2 = reshape(V(1:n*m), [n,m]);
    B2 = reshape(V(n*m+1:n*m+n), [n,1]);
    W3 = reshape(V(n*m+n+1:n*m+n+p*n), [p,n]);
    B3 = reshape(V(n*m+n+p*n+1:n*m+n+p*n+p), [p,1]);
    Xi = W2*X + B2;
    
    % computing output
    Z = logsig(Xi);
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
    Y_hat = W3*Z + B3;
    
elseif act_func==3 % radial basis function
    % decoding weights and biases of each layer
    W3 = reshape(V(1:p*n), [p,n]);
    B3 = reshape(V(p*n+1:p*n+p), [p,1]);
    
    if n==1
        if m==1
            c = (X_ramge(2)-X_range(1))/2;
            w = 2*c;
            Z = exp(-(X-c)' * (w^(-2)) * (X-c));
        elseif m==2
            c = [(X_ramge(1,2)-X_range(1,1))/2; (X_ramge(2,2)-X_range(2,1))/2];
            w = 2*c;
            Z = exp(-(X-c)' * diag(w.^(-2)) * (X-c));
        end
        
    else
        if m==1
            w = (X_range(2)-X_range(1))/(n-1);
            c = linspace(X_range(1), X_range(2), n); % centers of RBFs
            Z = zeros(n,1);
            for i=1 : n
                Z(i) = exp(-(X-c(i))' * (w^(-2)) * (X-c(i)));
            end 
            
        elseif m==2
            n_new = ceil(sqrt(n));
            w = (X_range(:,2)-X_range(:,1))/(n_new-1);
            [cx,cy] = meshgrid(linspace(X_range(1,1), X_range(1,2), n_new),...
                linspace(X_range(2,1), X_range(2,2), n_new));
            c = [cx(:)'; cy(:)']; % centers of RBFs
            n_new = n_new^2;
            Z = zeros(n_new,1);
            for i=1 : n_new
                Z(i) = exp(-(X-c(:,i))' * diag(w.^(-2)) * (X-c(:,i)));
            end
            %W3 = reshape(V(size(V,1)-p-p*n_new+1:size(V,1)-p), [p,n])
        end
    end
       
    Y_hat = W3*Z + B3;
end

end