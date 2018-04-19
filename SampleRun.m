%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% data generation

func_sel = 2; % funtion to be approximated
noise = 0; % noise level
N_sample = 500; % number of data points to be sampled 
ratio = 0.2; % ratio of testing set to whole data

GenerateData(func_sel,noise,N_sample); % function call to sample data
DivideData(ratio); % function call to split data into training and testing

%% reading data

fid = fopen('patterns.trn','rt'); % open training data file
fline = fgets(fid);
n_col = numel(strfind(fline,'x')) + numel(strfind(fline,'y')) + ...
    numel(strfind(fline,'z')); % determines number of inputs+outputs

if n_col==2
    % reading the training data
    data = textscan(fid,'%f %f');
    data = [cell2mat(data(1))'; cell2mat(data(2))'];
    fclose(fid);
    
    % reading the testing data
    fid = fopen('patterns.tst','rt');
    fline = fgets(fid);
    data_test = textscan(fid,'%f %f');
    data_test = [cell2mat(data_test(1))'; cell2mat(data_test(2))'];
    fclose(fid);
else
    % reading the training data
    data = textscan(fid,'%f %f %f');
    data = [cell2mat(data(1))'; cell2mat(data(2))'; cell2mat(data(3))'];
    fclose(fid);
        
    % reading the testing data
    fid = fopen('patterns.tst','rt');
    fline = fgets(fid);
    data_test = textscan(fid,'%f %f %f');
    data_test = [cell2mat(data_test(1))'; cell2mat(data_test(2))'; cell2mat(data_test(3))'];
    fclose(fid);
end

%% learning weights and biases of the network

m = 1; % number of inputs
n = 10; % number of hidden neurons
p = 1; % number of outputs

act_func = 1; % determines activation function of hidden layer

% training the neural net
[V,J_history,J_his_test] = Learning_VGD(m,n,p,data,data_test,act_func);

%% writing the weights to a file

    fid = fopen('network.wts','wt');
    fprintf(fid,'%3d, %3d, %3d\r\n',[m,n,p]);
    fprintf(fid,'%6.6f\r\n',V);
    fclose(fid);

%% decoding weights and biases

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