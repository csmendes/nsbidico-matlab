%This module implemets the data preprocessing and prediction functions
%   Parameters
%   <AllPopulationHistory> contains all the solutions from all generations
%   <NondominatedHistory>  contains the nondominated solutions from a generetion (if exists).
%   <time_horizon>               the number of generations ahead the model must predict
%   <neighbors>            the total number of past solutions used as closest solution
%   <currentGeneration>    current generation
%   <M>                    Number of problem objectives 

function [PredictedFront, Offsprings] = OSP(AllPopulationHistory, NondominatedHistory, ...
    time_horizon, neighbors, currentGeneration, Problem)
     
 
    metric  = {'euclidean', 'cityblock'};
    
    Data    = IdentifyClosestAllPop(AllPopulationHistory, NondominatedHistory, ...
        neighbors, currentGeneration, metric{1});
    
    Offsprings=[];
    predtype = 2;
   
    if predtype == 0 
        dataset = CreateDatasetForTS(Data, neighbors, Problem.M);
        PredictedFront = cell(length(dataset), 1);
    
        % for timeseries
        for i=1:length(dataset)
            TrainData = dataset{i};       
            model     = trainTimeSeriesModel(TrainData, Problem.M);
            predf     = forecast(model, TrainData, time_horizon);
            
            % get the points at the time <time_horizon> ahead
            predObjs          = predf.y(end,:); 
            PredictedFront{i} = num2cell(predObjs);
        end

    elseif predtype == 1

        %for regression
        dataset = CreateDataset(Data, neighbors, Problem.M);
        PredictedFront = cell(length(dataset), 1);
        for i=1:length(dataset)
            predObjs = {};
            
            for k=1:Problem.M            
                cols = {'generation', char("objective_" + char(string(k)))};    
                TrainData = dataset{i};
                TrainData = TrainData(:, cols);
                
                % for linear regression
                [trainedModel, validationRMSE] = trainRegressionModel(TrainData);
                predObjs{k} = trainedModel.predictFcn(array2table([time_horizon + currentGeneration], ...
                    'VariableNames',{'generation'}));
                                
            end
            PredictedFront{i} = predObjs;
        end

    elseif predtype == 2 
        % for multivariate regression
        dataset = CreateDatasetForMVR(Data, neighbors);
        PredictedFront = cell(length(dataset), 1);
        predObjs = {};
        for i=1:length(dataset)
            TrainingData = dataset{i};
            [TrainingData, mu, sigma] = FitTransform(TrainingData);

            Ytrain = TrainingData(:, [2:Problem.M+1]);
            [n,d] = size(Ytrain);
            xtrain = TrainingData(:,1);
            
            [beta, ~] = MVRegression(xtrain, Ytrain, n, d);
            sc_time_horizon = Transform(time_horizon + currentGeneration, mu(1), sigma(1));

            if ~isnan(beta)
                %fit = PredictMVR(beta, time_horizon + currentGeneration, d);
                fit = PredictMVR(beta, sc_time_horizon, d);
                fit = InverseTransform(fit, mu(2:Problem.M+1), sigma(2:Problem.M+1));
                for k=1:Problem.M
                    predObjs{k} = fit(k);
                end
            else
                for k=1:Problem.M
                    predObjs{k} = Inf;
                end
            end
            PredictedFront{i} = predObjs;
        end
        
    end
end

function [Data, mu, sigma] = FitTransform(Data)
    % 0 z-score for sample data; 1 z-score for populational data
    [Data, mu, sigma] = zscore(Data, 1); 

end

function Data = InverseTransform(Data, mu, sigma)
    Data = Data .* sigma + mu ;
end

function Data = Transform(Data, mu, sigma)
    Data = (Data - mu) / sigma;
end

function fits = PredictMVR(beta, data, d)
        
    B = [beta(1:d)'; repmat(beta(end), 1, d)];    
    fits = [ones(size(data)), data] * B;

end

function Offspring = InverseModeling(Problem,Population,PredictedFront,L)
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

    for i=1: length(PredictedFront)
        values = PredictedFront{i,1};
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

function Data = IdentifyClosest(AllPopulationHistory, NondominatedHistory, ...
    neighbors, currentGeneration, metric)

    Data = cell(length(NondominatedHistory{currentGeneration}), neighbors);    
    iterations = size(NondominatedHistory{currentGeneration}, 2);

    Current  = NondominatedHistory{currentGeneration};
    CurrentObjs = Current.objs;
    
    % normalize data in the interval [0,1]
    CurrentObjs = normalize(CurrentObjs, 'range');

    for i = 1:iterations 
        currentObj = CurrentObjs(i, :);
        j = 1;
        for gen = currentGeneration : -1: (currentGeneration - (neighbors - 1))
            Previous     = NondominatedHistory{gen - 1};  
            PreviousObjs = Previous.objs;                   
            
            % normalize data in the interval [0,1]
            PreviousObjs = normalize(PreviousObjs, 'range');
                     
            % select the closest solution
            [distance, closest] = pdist2(PreviousObjs, currentObj, metric, 'Smallest', 1);
            pos      = randi(length(distance), 1);
            distance = distance(pos);
            closest  = closest(pos);

            Data{i, j}.gen      = gen;
            Data{i, j}.distance = distance;
            Data{i, j}.closest  = NondominatedHistory{gen - 1}(closest);
            currentObj          = PreviousObjs(closest, :);
            j = j + 1;
        end
    end
end

function Data = IdentifyClosestAllPop(AllPopulationHistory, NondominatedHistory, ...
    neighbors, currentGeneration, metric)

    Current  = NondominatedHistory{currentGeneration};
       
    t = 1;
    while size(Current, 2) < 10
        Current = [Current, NondominatedHistory{currentGeneration - t}];
        t = t + 1;
    end

    CurrentObjs = Current.objs;

    % normalize data in the interval [0,1]
    CurrentObjsTemp = CurrentObjs;
    CurrentObjsTemp = normalize(CurrentObjs, 'range');
    
    SelectedPositions = [];
    
    Data = cell(length(CurrentObjsTemp), neighbors);    
    iterations = length(CurrentObjsTemp);

    for i = 1:iterations 
        currentObj = CurrentObjsTemp(i, :);
        j = 1;
        
        for gen = currentGeneration : -1: (currentGeneration - (neighbors - 1))
            Previous     = AllPopulationHistory{gen - 1};  
            PreviousObjs = Previous.objs;                   
            
            % normalize data in the interval [0,1]
            PreviousObjs = normalize(PreviousObjs, 'range');
                     
            if i > 1
                % select the closest solution
                [distance, closest] = pdist2(PreviousObjs, currentObj, metric, 'Smallest', 5);
                
                selj = SelectedPositions(:, j);

                % if none of the closest solutions have been selected
                if sum(ismember(selj, closest)) == 0
                    closest = closest(1);
                    SelectedPositions(i, j) = closest;
                    distance = distance(1);

                % if all the closest solutions already have been selected
                elseif sum(ismember(selj, closest)) >= 5
                    pos = randi(5, 1);
                    closest = closest(1);
                    SelectedPositions(i, j) = closest;
                    distance = distance(pos);

                % if at least one solution have not been selected
                else
                    pos = find(~ismember(closest, selj));
                    
                    if isempty(pos)
                        disp('Some shit going on');
                    end

                    closest = closest(pos(1));
                    SelectedPositions(i, j) = closest;
                    distance = distance(pos(1));
                end
            else
                % select the closest solution
                [distance, closest] = pdist2(PreviousObjs, currentObj, metric, 'Smallest', 1);
                SelectedPositions(i, j) = closest;
                distance = distance;                
            end

            Data{i, j}.gen      = gen;
            Data{i, j}.distance = distance;
            Data{i, j}.closest  = AllPopulationHistory{gen - 1}(closest);
            currentObj          = PreviousObjs(closest, :);
            j = j + 1;
        end
    end
end

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

function dataset = CreateDatasetForMVR(Data, neighbors)
       
    dataset = {};
    for i=1:size(Data, 1)
       
        Gens = [];
        Distances = [];
        Objs = [];
        for j=1:neighbors
            Gens = [Gens; Data{i,j}.gen];
            Objs = [Objs; Data{i,j}.closest.objs];            
        end
        dataset{i} = [Gens, Objs];
        dataset{i} = sortrows(dataset{i}, 1);
        
    end    
    %dataset = sortrows(dataset, 1);
    %dataset = array2table(dataset, 'VariableNames', headers);
end

function dataset = CreateDataset(Data, neighbors, M)
   
    headers = {'generation', 'distance'};    
    for i=1:M
        headers{length(headers)+1} = char("objective_" + char(string(i)));
    end
    dataset = {};
    for i=1:size(Data, 1)
        posdata = [];
        Gens = [];
        Distances = [];
        Objs = [];
        for j=1:neighbors
            Gens = [Gens; Data{i,j}.gen];
            Distances = [Distances; Data{i,j}.distance];
            Objs = [Objs; Data{i,j}.closest.objs];            
        end
        dataset{i} = [Gens, Distances, Objs];
        dataset{i} = sortrows(dataset{i}, 1);
        dataset{i} = array2table(dataset{i}, 'VariableNames', headers);
    end    
    %dataset = sortrows(dataset, 1);
    %dataset = array2table(dataset, 'VariableNames', headers);
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

function model = trainTimeSeriesModel(trainingData, M)
    
    na = eye(M);
    nb = [];
    nk = [];
    
    model = arx(trainingData, [na nb nk]);

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