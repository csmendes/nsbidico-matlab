function status = trigger(NondominatedSolutions, M)
    
    average = cell(length(NondominatedSolutions), 1);

    for i=1:length(NondominatedSolutions)
        solutions = NondominatedSolutions{i};
        Objs = [];
        for j=1:length(solutions)
            a = solutions(j).objs;
            Objs = [Objs; a];             
        end
        average{i} = mean(Objs); 
    end
    teste = 0;
end