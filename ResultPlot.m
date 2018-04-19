%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% reading the sampled data

n_tr = ceil(N_sample*(1-ratio)); % number of training data points
X_sample = zeros(m,n_tr);
Y_sample = zeros(p,n_tr);

for i=1 : n_tr
    X = data(1:m,i); % reading input 'k'
    Y = data(m+1:m+p,i); % reading output 'k'
    X_sample(:,i) = X; Y_sample(:,i) = Y;
end

X_range = [min(data(1:m,:),[],2), max(data(1:m,:),[],2)];

%% computing NN output funtion and plotting real and approximated functions

if func_sel==1
    [x,y] = meshgrid(linspace(X_range(1,1), X_range(1,2), 50),...
        linspace(X_range(1,1), X_range(1,2), 50));
    z = zeros(size(x));
    z_est = zeros(size(x));
    
    for i=1 : size(x,1)
        for j=1 : size(x,2)
            z_est(i,j) = NeuralNet(m,n,p,act_func,[x(i,j); y(i,j)],X_range,V) ;
        end
    end
    
    figure;
    plot3([0 0 1 1],[0 1 0 1],[0 1 1 0],'k.','MarkerSize',15);
    colorbar;
    hold on;
    plot3(X_sample(1,:),X_sample(2,:),Y_sample(1,:),'r.','MarkerSize',15);
    surf(x,y,z_est);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Function Approximation by ANN');
    legend('Original function','Sampled points','Estimated function');
    set(gca,'YMinorTick','on')
    grid on;
    grid minor;
    
elseif func_sel==2
    x = linspace(-2, 2, 2500);
    z = zeros(size(x));
    z_est = zeros(size(x));
    
    for i=1 : size(x,2)
        z(i) = sinexp(x(i));
        z_est(i) = NeuralNet(m,n,p,act_func,x(i),X_range,V);
    end
    
    figure;
    plot(x(:),z(:),'k-','linewidth',0.5);
    hold on;
    plot(X_sample,Y_sample,'r.','MarkerSize',10);
    plot(x(:),z_est(:),'k:','linewidth',1.5);
    xlabel('X');
    ylabel('Y');
    title('Function Approximation by ANN');
    legend('Original function','Sampled points','Estimated function');
    set(gca,'YMinorTick','on')
    grid on;
    grid minor
    
elseif func_sel==3
    [x,y] = meshgrid(linspace(X_range(1,1), X_range(1,2), 50),...
        linspace(X_range(1,1), X_range(1,2), 50));
    z = zeros(size(x));
    z_est = zeros(size(x));
    
    for i=1 : size(x,1)
        for j=1 : size(x,2)
            z(i,j) = sphere_one(x(i,j),y(i,j));
            z_est(i,j) = NeuralNet(m,n,p,act_func,[x(i,j); y(i,j)],X_range,V) ;
        end
    end
    
    figure;
    mesh(x,y,z);
    colorbar;
    hold on;
    hidden off;
    plot3(X_sample(1,:),X_sample(2,:),Y_sample(1,:),'r.','MarkerSize',15);
    surf(x,y,z_est);
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Function Approximation by ANN');
    legend('Original function','Sampled points','Estimated function');
    set(gca,'YMinorTick','on')
    grid on;
    grid minor;
    
elseif func_sel==4
    x = linspace(-2, 2, 2500);
    z = zeros(size(x));
    z_est = zeros(size(x));
    
    for i=1 : size(x,2)
        z(i) = sinc(3*x(i));
        z_est(i) = NeuralNet(m,n,p,act_func,x(i),X_range,V);
    end
    
    figure;
    plot(x(:),z(:),'k-','linewidth',0.5);
    hold on;
    plot(X_sample,Y_sample,'r.','MarkerSize',10);
    plot(x(:),z_est(:),'k:','linewidth',1.5);
    xlabel('X');
    ylabel('Y');
    title('Function Approximation by ANN');
    legend('Original function','Sampled points','Estimated function');
    set(gca,'YMinorTick','on')
    grid on;
    grid minor;
end

%% plotting cost functions

figure;
plot(J_his_test,'k:')
hold on;
plot(J_history,'color','k')
set(gca,'YMinorTick','on')
grid on;
grid minor;
xlabel('Epochs');
ylabel('Cost Function (MSE)');
title('Function Approximation by ANN');
legend('Training data','Testing data');

set(groot,'defaultAxesFontName', 'Times');
set(groot,'DefaultFigureColormap',summer);