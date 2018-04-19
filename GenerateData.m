%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function GenerateData(func_sel,noise,Np)
% Generate data
% This function samples data from a given function
% Inputs:
%   func_sel -> original function to generate data from
%   noise -> noise level to add to the data
%   Np -> number of entire data points
% Outputs:
%   one text file for data without noise and
%   one text file for data with noise is created

if func_sel==1
    data_wo = [...
        0 0 1 1;...
        0 1 0 1;...
        0 1 1 0]; % XOR data
    rand_ind = randi(4,1,Np);
    noise = 0.1*randn(3,Np);
    data = zeros(3,Np);
    for i=1 : Np
        data(:,i) = data_wo(:,rand_ind(i)) + noise(:,i);
    end
    
elseif func_sel==2
    x_min = -2; x_max = 2; % input range
    x = linspace(x_min, x_max, Np); % input points
    data_wo = zeros(2,Np); % data without noise
    for i=1 : Np
        data_wo(:,i) = [x(i),sinexp(x(i))];
    end
    data = data_wo + noise*randn(2,Np);
    
elseif func_sel==3
    x_min = -2; x_max = 2; % input range
    y_min = -2; y_max = 2; % input range
    Nxy = ceil(sqrt(Np)); % number of points on each x and y axis
    Np = Nxy^2; % actual number of data points
    x = linspace(x_min, x_max, Nxy);
    y = linspace(y_min, y_max, Nxy);
    k = 1;
    data_wo = zeros(3,Np); % data without noise
    for i=1 : Nxy
        for j=1 : Nxy
            data_wo(:,k) = [x(i),y(j),sphere_one(x(i),y(j))];
            k = k+1;
        end
    end
    data = data_wo + noise*randn(3,Np);
    
elseif func_sel==4
    x_min = -2; x_max = 2; % input range
    x = linspace(x_min, x_max, Np); % input points
    data_wo = zeros(2,Np); % data without noise
    for i=1 : Np
        data_wo(:,i) = [x(i),sinc(3*x(i))];
    end
    data = data_wo + noise*randn(2,Np);
    
end % end of "if" for function selection

write(func_sel,data_wo,data);

end

function write(func_sel,data_wo,data)
% This function writes the generated data to text files
% Inputs:
%   func_sel -> original function to generate data from
%   data_wo -> data without noise
%   data -> data with noise

if func_sel==2 || func_sel==4 % case with one input x, one output y
    fid = fopen('DataWithoutNoise.txt','wt');
    fprintf(fid,'%12s %12s\n','x','y');
    fprintf(fid,'%12.6f %12.6f\n',data_wo);
    fclose(fid);
    
    fid = fopen('DataWithNoise.txt','wt');
    fprintf(fid,'%12s %12s\n','x','y');
    fprintf(fid,'%12.6f %12.6f\n',data);
    fclose(fid);
elseif func_sel==1 || func_sel==3 % case with two inputs x & y, one output z
    fid = fopen('DataWithoutNoise.txt','wt');
    fprintf(fid,'%12s %12s %12s\n','x','y','z');
    fprintf(fid,'%12.6f %12.6f %12.6f\n',data_wo);
    fclose(fid);
    
    fid = fopen('DataWithNoise.txt','wt');
    fprintf(fid,'%12s %12s %12s\n','x','y','z');
    fprintf(fid,'%12.6f %12.6f %12.6f\n',data);
    fclose(fid);
end

end