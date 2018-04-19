%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% opening the file containing weights
fid = fopen('network.wts','rt');
fline = fgets(fid);
neurons = str2double(regexp(fline,'\d*','match')');

% extracting the number of neurons at each layer
m = neurons(1);
n = neurons(2);
p = neurons(3);

% reading the concatenated weights
V = textscan(fid,'%f');
V = cell2mat(V)';
fclose(fid);

fid = fopen('patterns.tst','rt'); % open testing data file
fline = fgets(fid);

if m==1
    % reading the testing data
    data_test = textscan(fid,'%f %f');
    data_test = [cell2mat(data_test(1))'; cell2mat(data_test(2))'];
    fclose(fid);
else   
    % reading the testing data
    data_test = textscan(fid,'%f %f %f');
    data_test = [cell2mat(data_test(1))'; cell2mat(data_test(2))'; cell2mat(data_test(3))'];
    fclose(fid);
end

% giving the test data to the NN
X_range = 4*ones(size(data_test,1),1);
output_test = zeros(1,size(data_test,2));
for i=1 : size(data_test,2)
   output_test(i) = NeuralNet(m,n,p,act_func,data_test(1:m,i),X_range,V);
end

% plotting actual data and estimated data
if m==1
    figure;
    plot(data_test(1,:),data_test(2,:),'k-','linewidth',0.5);
    hold on;
    plot(data_test(1,:),output_test(:),'k:','linewidth',1.5);
    xlabel('X');
    ylabel('Y');
    title('Testing ANN Function');
    legend('Original data','Estimated data');
    set(gca,'YMinorTick','on')
    grid on;
    grid minor
else
    figure;
    plot3(data_test(1,:),data_test(2,:),data_test(3,:),'k.','MarkerSize',15);
    hold on;
    plot3(data_test(1,:),data_test(2,:),output_test(:),'ks','MarkerSize',5);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Testing ANN Function');
    legend('Original data','Estimated data');
    set(gca,'YMinorTick','on')
    grid on;
    grid minor;
end

% calculating the cost function
J_test = 1/size(data_test,2)*norm(data_test(end,:)-output_test(:)')^2;
display(J_test);