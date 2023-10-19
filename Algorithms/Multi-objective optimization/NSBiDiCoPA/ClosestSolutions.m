function Solutions = ClosestSolutions(PredictedFront,Samples)
    
    DSPObjs = Samples.objs;
    OSPObjs = [];
    for i=1:length(PredictedFront)
        values = PredictedFront{i,1};
        OSPObjs(i,1) = values{1};
        OSPObjs(i,2) = values{2};
    end
    
    Indx = [];
    for i = 1:length(OSPObjs)
        [~, closest] = pdist2(DSPObjs, OSPObjs(i,:), 'euclidean', 'Smallest', 1);
        pos      = randi(length(closest), 1);
        closest  = closest(pos);
        Indx(i) = closest;        
    end
    Solutions = Samples(unique(Indx));

end
