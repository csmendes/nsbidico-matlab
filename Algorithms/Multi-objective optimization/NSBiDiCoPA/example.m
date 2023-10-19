clear all;clc

data=load('abalone_dataset');
X=zscore(data.abaloneInputs);
Y=zscore(data.abaloneTargets);

for i=1:100

[model, Stats1]=elm_train(X',Y', 0, i, 'rbf');
[performance(i)] = [Stats1.TrainingAccuracy];

end

plot(performance,'LineWidth', 2);
legend('training');
xlabel('hidden nodes')
ylabel('RMSE')
grid

%##########################################################
%##########################################################
%##########################################################

% load('training', 'TrainingData');
% TrainingData = zscore(TrainingData);
% Training = TrainingData(1:70,:);
% Test     = TrainingData(71:end, : );

% Ytrain = Training(:, 4:end);
% xtrain = Training(:,1:3);
% 
% Ytest = Test(:, 4:end);
% xtest = Test(:,1:3);
% 
% for i=1:100
% 
% [model, Stats1]=elm_train(xtrain,Ytrain, 0, i, 'rbf');
% [~, Stats2] = elm_predict(model, xtest, Ytest);
% [performance1(i)] = [Stats1.TrainingAccuracy];
% [performance2(i)] = [Stats2.TestingAccuracy];
% end

% for i=1:200
%     [~, Stats]=elm_train(X',Y',0, i, 'rbf');
%     [performance(i)] = Stats.TrainingAccuracy;
% end

% plot(performance1,'LineWidth', 2);
% hold on;
% plot(performance2,'LineWidth', 2);
% legend('testing', 'training');
% xlabel('hidden nodes')
% ylabel('RMSE')
% grid