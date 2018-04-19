%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% applying MATLAB neural net toolbox

% reading the data
inputs = data(1:m,:);
targets = data(m+1:m+p,:);

% Create a Fitting Network
hiddenLayerSize = n;
net = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% training parameters
% net.trainParam.lr = 0.001; % uncomment for changing the learning rate
net.trainParam.max_fail=1000;
net.trainParam.epochs = 1000;

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);

% View the Network
view(net);

%% plotting

if func_sel==1
    [x,y] = meshgrid(linspace(X_range(1,1), X_range(1,2), 50),...
        linspace(X_range(1,1), X_range(1,2), 50));
    z = zeros(size(x));
    z_est = zeros(size(x));
    
    for i=1 : size(x,1)
        for j=1 : size(x,2)
            z_est(i,j) = net([x(i,j); y(i,j)]) ;
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
        z_est(i) = net(x(i));
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
            z_est(i,j) = net([x(i,j); y(i,j)]) ;
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
        z_est(i) = net(x(i));
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
plotperform(tr);

set(groot,'defaultAxesFontName', 'Times');
set(groot,'DefaultFigureColormap',summer);