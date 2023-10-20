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