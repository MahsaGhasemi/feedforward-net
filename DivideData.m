%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DivideData(ratio)
% Divide data
% This function divides data into training set and testing set
% Inputs:
%   ratio -> ratio of testing set to whole data---from [0,1]
% Outputs:
%   one text file for training set and 
%   one text file for testing set is created

% find the number of columns in the data file
fid = fopen('DataWithNoise.txt','rt');
fline = fgets(fid);
n_col = numel(strfind(fline,'x')) + numel(strfind(fline,'y')) + numel(strfind(fline,'z'));

% read the noisy data and store it in 'data'
if n_col==2
    data = textscan(fid,'%f %f');
    data = [cell2mat(data(1))'; cell2mat(data(2))'];
    fclose(fid);
elseif n_col==3
    data = textscan(fid,'%f %f %f');
    data = [cell2mat(data(1))'; cell2mat(data(2))'; cell2mat(data(3))'];
    fclose(fid);
end

[c,n] = size(data);
n_tr = ceil(n*(1-ratio)); % number of training data points
n_ts = n-n_tr; % number of testing data points
data_tr = zeros(c,n_tr);
data_ts = zeros(c,n_ts);

% separating training points and testing points
ind = [sort(randperm(n,n_ts)), 0]; % indices of the testing set
ind_tr = 1; % counter of training data points
ind_ts = 1; % counter of testing data points
for i=1 : n
    if i ~=ind(ind_ts)
        data_tr(:,ind_tr) = data(:,i);
        ind_tr = ind_tr+1;
    else
        data_ts(:,ind_ts) = data(:,i);
        ind_ts = ind_ts+1;
    end
    
end

write(n_col,data_tr,data_ts);

function write(n_col,data_tr,data_ts)
% This function writes the divided data to text files
% Inputs:
%   n_col -> number of inputs+outputs of data
%   data_tr -> training set
%   data_ts -> testing set

if n_col==2 % case with one input x, one output y
    fid = fopen('patterns.trn','wt');
    fprintf(fid,'%12s %12s\n','x','y');
    fprintf(fid,'%12.6f %12.6f\n',data_tr);
    fclose(fid);
    
    fid = fopen('patterns.tst','wt');
    fprintf(fid,'%12s %12s\n','x','y');
    fprintf(fid,'%12.6f %12.6f\n',data_ts);
    fclose(fid);
elseif n_col==3 % case with two inputs x & y, one output z
    fid = fopen('patterns.trn','wt');
    fprintf(fid,'%12s %12s %12s\n','x','y','z');
    fprintf(fid,'%12.6f %12.6f %12.6f\n',data_tr);
    fclose(fid);
    
    fid = fopen('patterns.tst','wt');
    fprintf(fid,'%12s %12s %12s\n','x','y','z');
    fprintf(fid,'%12.6f %12.6f %12.6f\n',data_ts);
    fclose(fid);
end

end % end of function 'write'

end