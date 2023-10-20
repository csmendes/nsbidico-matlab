function Data = IdentifyClosestAllPop(AllPopulationHistory, NondominatedHistory, neighbors, currentGeneration, metric)

    Current  = NondominatedHistory{currentGeneration};
       
    t = 1;
    while size(Current, 2) < 10
        Current = [Current, NondominatedHistory{currentGeneration - t}];
        t = t + 1;
    end

    CurrentObjs = Current.objs;

    % normalize data in the interval [0,1]
    CurrentObjsTemp = CurrentObjs;
    CurrentObjsTemp = normalize(CurrentObjsTemp, 'range');
    
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