%This module implemets the data preprocessing and prediction functions
%   Parameters
%   <AllPopulationHistory> contains all the solutions from all generations
%   <NondominatedHistory>  contains the nondominated solutions from a generetion (if exists).
%   <time_horizon>               the number of generations ahead the model must predict
%   <neighbors>            the total number of past solutions used as closest solution
%   <currentGeneration>    current generation
%   <M>                    Number of problem objectives 

function [PredictedDecs, Offsprings] = DSP(Data, time_horizon, neighbors, ...
    currentGeneration, Problem, PredictedOS)
     
    predtype = 3;

    if predtype == 0
        dataset = CreateDataset(Data, neighbors, Problem.D);
        PredictedDecs = zeros(length(dataset), Problem.D);
        predDecs = [];
        % for timeseries
        for i=1:length(dataset)
            TrainData = dataset{i};       
            model     = trainTimeSeriesModel(TrainData, Problem.D);
            predf     = forecast(model, TrainData, time_horizon);
            
            % get the points at the time <time_horizon> ahead
            predDecs         = [predDecs; predf.y(end,:)]; 
            PredictedDecs{i} = num2cell(predDecs);
        end
    
    elseif predtype == 1
        dataset = CreateDataset(Data, neighbors, Problem.D);
        predDecs = zeros(length(dataset), Problem.D);
        % for regression
        for i=1:length(dataset)
            
            TrainingData = dataset{i};
            for d=1:Problem.D            
                cols = {'generation', char("variable_" + char(string(d)))};    
                            
                % for linear regression
                [trainedModel, validationRMSE] = trainRegressionModel(TrainingData(:, cols));
                predDecs(i,d) = trainedModel.predictFcn(array2table( ...
                    [time_horizon + currentGeneration],'VariableNames',{'generation'}));
              
            end    
            PredictedDecs{i} = predDecs;
        end
    
    elseif predtype == 2

        % for multivariate regression

        dataset = CreateDatasetForMVR(Data, neighbors);
        predDecs = zeros(length(dataset), Problem.D);
        PredictedDecs = cell(length(dataset), 1);
        predDecs = [];
        
        for i=1:length(dataset)
            TrainingData = dataset{i};       
            %[TrainingData, mu, sigma] = FitTransform(TrainingData);

            Ytrain = TrainingData(:, [2+Problem.M:end]);
            [n,d] = size(Ytrain);
            xtrain = TrainingData(:,1:1+Problem.M);

            [beta, ~] = MVRegression(xtrain, Ytrain, n, d);
            %sc_time_horizon = Transform(time_horizon+currentGeneration, mu(1:1+Problem.M), sigma(1:1+Problem.M));

            if ~isnan(beta)
                fit = PredictMVR(beta, time_horizon + currentGeneration, d);
                
                %fit = PredictMVR(beta, sc_time_horizon, d);
                %fit = InverseTransform(fit, mu(2:Problem.D+1), sigma(2:Problem.D+1));
                %PredictedDecs{i} = num2cell(fit);
                predDecs = [predDecs; fit];
%                 for k=1:Problem.D
%                     predDecs{k} = fit(k);
%                 end
            else
                for k=1:Problem.D
                    predDecs(i,k) = Inf;
                end                
                % predDecs = [predDecs; fit]
                %PredictedDecs{i} = predDecs;
            end
            PredictedDecs{i} = predDecs;
        end
    
    
    elseif predtype == 3

        % for Extreme Learning Machine regression

        dataset = CreateDatasetForMVR(Data, neighbors);
        predDecs = zeros(length(dataset), Problem.D);
        PredictedDecs = cell(length(dataset), 1);
        predDecs = [];
        
        for i=1:length(dataset)
            TrainingData = dataset{i};       
            % [TrainingData, mu, sigma] = FitTransform(TrainingData);

            Ytrain = TrainingData(:, 2+Problem.M:end);
            [n,d] = size(Ytrain);
            xtrain = TrainingData(:, 1:1+Problem.M);

            [model, Stats] = elm_train(xtrain, Ytrain, 0, Problem.D, 'rbf');
            predobj = [PredictedOS{i}{1,1}, PredictedOS{i}{1,2}];
            X2Pred = [time_horizon+currentGeneration, predobj];
            %X2Pred = Transform(X2Pred, mu(1:1+Problem.M), sigma(1:1+Problem.M));
            
            fit = elm_predict(model, X2Pred, Inf);
            %fit = InverseTransform(fit, mu(2:Problem.D+1), sigma(2:Problem.D+1));
            
            %PredictedDecs{i} = num2cell(fit);
            predDecs = [predDecs; fit'];  
            PredictedDecs{i} = predDecs;
        end
        
    end

    % Convert invalid values to random values
    %[N,D]   = size(predDecs);
    %Lower   = repmat(Problem.lower,N,1);
    %Upper   = repmat(Problem.upper,N,1);
    %andDec = unifrnd(Lower,Upper);
    %invalid = predDecs<Lower | predDecs>Upper;
    %predDecs(invalid) = randDec(invalid);

    Offsprings = Problem.Evaluation(predDecs);

end

function [Data, mu, sigma] = FitTransform(Data)
    % 0 z-score for sample data; 1 z-score for populational data
    [Data, mu, sigma] = zscore(Data, 1); 

end

function Data = InverseTransform(Data, mu, sigma)
    Data = Data * sigma + mu ;
end

function Data = Transform(Data, mu, sigma)
    Data = (Data - mu) / sigma;
end

function fits = PredictMVR(beta, data, d)
        
    B = [beta(1:d)'; repmat(beta(end), 1, d)];    
    fits = [ones(size(data)), data] * B;

end



function Offspring = InverseModeling(Problem,Population,PredictedDecs,L)
% The Gaussian process based reproduction

% This function is modified from the code in
% http://www.soft-computing.de/jin-pub_year.html

    % Parameter setting
    if nargin < 4
        L = 3;
    end
    PopDec = Population.decs;
	PopObj = Population.objs;
    [N,D]  = size(PopDec);
    
    PredObj = [];

    for i=1: length(PredictedDecs)
        values = PredictedDecs{i,1};
        PredObj(i,1) = values{1};
        PredObj(i,2) = values{2};
    end

    % Gaussian process based reproduction
    if length(Population) < 2*Problem.M
        OffDec = PopDec;
    else
        OffDec = [];
%         fmin   = 1.5*min(PopObj,[],1) - 0.5*max(PopObj,[],1);
%         fmax   = 1.5*max(PopObj,[],1) - 0.5*min(PopObj,[],1);

        fmin   = 1.5*min(PredObj,[],1) - 0.5*max(PredObj,[],1);
        fmax   = 1.5*max(PredObj,[],1) - 0.5*min(PredObj,[],1);

        % Train one groups of GP models for each objective
        for m = 1 : Problem.M
            parents = randperm(N,floor(N/Problem.M));
            offDec  = PopDec(parents,:);
            for d = randperm(D,L)
                % Gaussian Process
                try
%                     [ymu,ys2] = gp(struct('mean',[],'cov',[],'lik',log(0.01)),...
%                                    @infExact,@meanZero,@covLIN,@likGauss,...
%                                    PopObj(parents,m),PopDec(parents,d),...
%                                    linspace(fmin(m),fmax(m),size(offDec,1))');

                    [ymu,ys2] = gp(struct('mean',[],'cov',[],'lik',log(0.01)),...
                                   @infExact,@meanZero,@covLIN,@likGauss,...
                                   PredObj(parents,m),PopDec(parents,d),...
                                   linspace(fmin(m),fmax(m),size(offDec,1))');
                    
                    offDec(:,d) = ymu + rand*sqrt(ys2).*randn(size(ys2));
                catch
                end
            end
            OffDec = [OffDec;offDec];
        end
    end
    
    % Convert invalid values to random values
    [N,D]   = size(OffDec);
    Lower   = repmat(Problem.lower,N,1);
    Upper   = repmat(Problem.upper,N,1);
    randDec = unifrnd(Lower,Upper);
    invalid = OffDec<Lower | OffDec>Upper;
    OffDec(invalid) = randDec(invalid);

    % Polynomial mutation
%     [proM,disM] = deal(1,20);
%     Site   = rand(N,D) < proM/D;
%     mu     = rand(N,D);
%     temp   = Site & mu<=0.5;
%     OffDec = min(max(OffDec,Lower),Upper);
%     OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
%                    (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
%     temp = Site & mu>0.5; 
%     OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
%                    (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    Offspring = Problem.Evaluation(OffDec);
end

% function Data = IdentifyClosest(AllPopulationHistory, NondominatedHistory, ...
%     neighbors, currentGeneration, metric)

%     Data = cell(length(NondominatedHistory{currentGeneration}), neighbors);    
%     iterations = size(NondominatedHistory{currentGeneration}, 2);

%     Current  = NondominatedHistory{currentGeneration};
%     CurrentDecs = Current.decs;
%     % ZminC = min(Current.decs, [], 1);
%     % CurrentDecs = (Current.decs-repmat(ZminC,length(Current.decs),1))./(repmat(max(Current.decs),...
%     %                 length(Current.decs),1)-repmat(ZminC,length(Current.decs),1)+1e-10)+1e-10;

%     for i = 1:iterations 
%         currentDec = CurrentDecs(i, :);
%         j = 1;
%         for gen = currentGeneration : -1: (currentGeneration - (neighbors - 1))
%             Previous     = NondominatedHistory{gen - 1};  
%             PreviousDecs = Previous.decs;         
%             % ZminP = min(Previous.decs, [], 1);
%             % PreviousDecs = (Previous.decs-repmat(ZminP,length(Previous.decs),1))./(repmat(max(Previous.decs),...
%             %                 length(Previous.decs),1)-repmat(ZminP,length(Previous.decs),1)+1e-10)+1e-10;
                     
%             % select the closest solution
%             [distance, closest] = pdist2(PreviousDecs, currentDec, metric, 'Smallest', 5);
%             pos      = randi(length(distance), 1);
%             distance = distance(pos);
%             closest  = closest(pos);

%             Data{i, j}.gen      = gen;
%             Data{i, j}.distance = distance;
%             Data{i, j}.closest  = NondominatedHistory{gen - 1}(closest);
%             currentDec          = PreviousDecs(closest, :);
%             j = j + 1;
%         end
%     end
% end


function dataset = CreateDatasetForTS(Data, neighbors, M)
   
    dataset = cell(size(Data, 1), 1);
    for i=1:size(Data, 1)
        Gens = [];
        Objs = [];
        for j=1:neighbors
            Gens = [Gens; Data{i,j}.gen];
            % Distances = [Distances; Data{i,j}.distance];
            Objs = [Objs; Data{i,j}.closest.objs];            
        end
        temp_data = [Gens, Objs];
        temp_data = sortrows(temp_data, 1);
        temp_data = temp_data(:,[1:M]+1);
        
        y = temp_data(all(~isnan(temp_data), 2), :);
        [~, idx_y] = unique(y(:, 1), 'stable');
        
        data = y(idx_y, :); 
        
        if size(data, 1) < 3
            data = y;
        end
        
%         data = temp_data;

        data = iddata(data, [], 'TimeUnit', 'hours');
        dataset{i} = data;
    end    
  
end

function dataset = CreateDataset(Data, neighbors, D)
   
    headers = {'generation', 'distance'};    
    for i=1:D
        headers{length(headers)+1} = char("variable_" + char(string(i)));
    end
    dataset = {};
    for i=1:size(Data, 1)
        posdata = [];
        Gens = [];
        Distances = [];
        Decs = [];
        for j=1:neighbors
            Gens = [Gens; Data{i,j}.gen];
            Distances = [Distances; Data{i,j}.distance];
            Decs = [Decs; Data{i,j}.closest.decs];            
        end
        dataset{i} = [Gens, Distances, Decs];
        dataset{i} = sortrows(dataset{i}, 1);
        dataset{i} = array2table(dataset{i}, 'VariableNames', headers);
    end    
    %dataset = sortrows(dataset, 1);
    %dataset = array2table(dataset, 'VariableNames', headers);
end

function dataset = CreateDatasetForMVR(Data, neighbors)
       
    dataset = {};
    for i=1:size(Data, 1)
       
        Gens = [];
        Distances = [];
        Decs = [];
        Objs = [];
        for j=1:neighbors
            Gens = [Gens; Data{i,j}.gen];
            Objs = [Objs; Data{i,j}.closest.objs];
            Decs = [Decs; Data{i,j}.closest.decs];            
        end
        dataset{i} = [Gens, Objs, Decs];
        dataset{i} = sortrows(dataset{i}, 1);
        
    end    
    %dataset = sortrows(dataset, 1);
    %dataset = array2table(dataset, 'VariableNames', headers);
end

function model = trainTimeSeriesModel(trainingData, M)
    
    na = eye(M);
    nb = [];
    nk = [];
    
    model = arx(trainingData, [na nb nk]);

end

function [beta, sigma] = MVRegression(xtrain, Ytrain, n, d)
   
    % multivariate linear regression

    % Ytrain = zscore(Ytrain);
    % xtrain = zscore(xtrain);
    
    
    Xtrain = cell(n,1);
    for i=1:n
        Xtrain{i} = [eye(d) repmat(xtrain(i), d, 1)];
    end
    try
        [beta, sigma] = mvregress(Xtrain,Ytrain);
    catch
        beta = NaN;
        sigma = NaN;
    end
end

function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
    % [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
    % Returns a trained regression model and its RMSE. This code recreates the
    % model trained in Regression Learner app. Use the generated code to
    % automate training the same model with new data, or to learn how to
    % programmatically train models.
    %
    %  Input:
    %      trainingData: A table containing the same predictor and response
    %       columns as those imported into the app.
    %
    %  Output:
    %      trainedModel: A struct containing the trained regression model. The
    %       struct contains various fields with information about the trained
    %       model.
    %
    %      trainedModel.predictFcn: A function to make predictions on new data.
    %
    %      validationRMSE: A double containing the RMSE. In the app, the Models
    %       pane displays the RMSE for each model.
    %
    % Use the code to train the model with new data. To retrain your model,
    % call the function from the command line with your original data or new
    % data as the input argument trainingData.
    %
    % For example, to retrain a regression model trained with the original data
    % set T, enter:
    %   [trainedModel, validationRMSE] = trainRegressionModel(T)
    %
    % To make predictions with the returned 'trainedModel' on new data T2, use
    %   yfit = trainedModel.predictFcn(T2)
    %
    % T2 must be a table containing at least the same predictor columns as used
    % during training. For details, enter:
    %   trainedModel.HowToPredict
       
    
    % Extract predictors and response
    % This code processes the data into the right shape for training the
    % model.
    inputTable = trainingData;
    %predictorNames = {'TimeOfDay'};
    predictorNames = {'generation'};
    predictors = inputTable(:, predictorNames);
    %response = inputTable.Duration;
    response = inputTable(:, end);
    isCategoricalPredictor = [false];
    
    % Train a regression model
    % This code specifies all the model options and trains the model.
    % concatenatedPredictorsAndResponse = predictors;
    % concatenatedPredictorsAndResponse = response;
    linearModel = fitlm([predictors, response], ...
        'linear', 'RobustOpts', 'off');
    
    % Create the result struct with predict function
    predictorExtractionFcn = @(t) t(:, predictorNames);
    linearModelPredictFcn = @(x) predict(linearModel, x);
    trainedModel.predictFcn = @(x) linearModelPredictFcn(predictorExtractionFcn(x));
    
    % Add additional fields to the result struct
    %trainedModel.RequiredVariables = {'TimeOfDay'};
    trainedModel.RequiredVariables = {'generation'};
    trainedModel.LinearModel = linearModel;
    trainedModel.About = 'This struct is a trained model exported from Regression Learner R2022b.';
    trainedModel.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');
    
    % Extract predictors and response
    % This code processes the data into the right shape for training the
    % model.
    inputTable = trainingData;
    %predictorNames = {'TimeOfDay'};
    predictorNames = {'generation'};
    predictors = inputTable(:, predictorNames);
    %response = inputTable.Duration;
    response = inputTable(:, end);
    isCategoricalPredictor = [false];
    
    validationPredictFcn = @(x) linearModelPredictFcn(x);
    
    % Compute resubstitution predictions
    validationPredictions = validationPredictFcn(predictors);
    response = response{:,1};
    % Compute validation RMSE
    isNotMissing = ~isnan(validationPredictions) & ~isnan(response);
    validationRMSE = sqrt(nansum(( validationPredictions - response ).^2) / numel(response(isNotMissing) ));
end