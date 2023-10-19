%This module implemets the data preprocessing and prediction functions
%   Parameters
%   <AllPopulationHistory> contains all the solutions from all generations
%   <NondominatedHistory>  contains the nondominated solutions from a generetion (if exists).
%   <lambda>               the number of generations ahead the model must predict
%   <neighbors>            the total number of past solutions used as closest solution
%   <currentGeneration>    current generation
%   <M>                    Number of problem objectives 

function PredictedFront = OSP(AllPopulationHistory, NondominatedHistory, lambda, ...
    neighbors, currentGeneration, M)
    
    
    
    models  = {};
    metric  = {'euclidean', 'cityblock'};
    Data    = IdentifyClosest(AllPopulationHistory, NondominatedHistory, neighbors, currentGeneration, metric{1});
    dataset = CreateDataset(Data, neighbors, M);
    PredictedFront = cell(length(dataset), 1);

    for i=1:length(dataset)
        predObjs = {};
        for k=1:M            
            cols = {'generation', char("objective_" + char(string(k)))};    
            TrainData = dataset{i};
            TrainData = TrainData(:, cols);

            [trainedModel, validationRMSE] = trainRegressionModel(TrainData);
            predObjs{k} = trainedModel.predictFcn(array2table([lambda + currentGeneration], 'VariableNames',{'generation'}));
        end
        PredictedFront{i} = predObjs;
    end
    a = 0;
end


function Data = IdentifyClosest(AllPopulationHistory, NondominatedHistory, ...
    neighbors, currentGeneration, metric)

    Data = cell(length(NondominatedHistory{currentGeneration}), neighbors);    
    iterations = size(NondominatedHistory{currentGeneration}, 2);

    Current  = NondominatedHistory{currentGeneration};
    CurrentObjs = Current.objs;
    % ZminC = min(Current.objs, [], 1);
    % CurrentObjs = (Current.objs-repmat(ZminC,length(Current.objs),1))./(repmat(max(Current.objs),...
    %                 length(Current.objs),1)-repmat(ZminC,length(Current.objs),1)+1e-10)+1e-10;

    for i = 1:iterations 
        currentObj = CurrentObjs(i, :);
        j = 1;
        for gen = currentGeneration : -1: (currentGeneration - (neighbors - 1))
            Previous     = NondominatedHistory{gen - 1};  
            PreviousObjs = Previous.objs;         
            % ZminP = min(Previous.objs, [], 1);
            % PreviousObjs = (Previous.objs-repmat(ZminP,length(Previous.objs),1))./(repmat(max(Previous.objs),...
            %                 length(Previous.objs),1)-repmat(ZminP,length(Previous.objs),1)+1e-10)+1e-10;
                     
            % select the closest soluction
            [distance, closest] = pdist2(PreviousObjs, currentObj, metric, 'Smallest', 4);
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
        'quadratic', 'RobustOpts', 'off');
    
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