classdef NSBiDiCoPA < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained>
% Bidirectional coevolution constrained multiobjective evolutionary algorithm

%------------------------------- Reference --------------------------------
% Z. Liu, B. Wang, and K. Tang, Handling constrained multiobjective
% optimization problems via bidirectional coevolution, IEEE Transactions on
% Cybernetics, 2021.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2022 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

	methods
		function main(Algorithm,Problem)
			
			% Parameters pool
			% CrPool = [0.1, 0.2, 0.5, 0.9];
			% FPool  = [0.4, 0.6, 0.8, 0.9];

			[Cr, F, proM, disM] = Algorithm.ParameterSet(0.95, 0.5, 1, 20);


			% Generate random population
			Population = Problem.Initialization();
			ArcPop     = [];
			
            NondominatedHistory   = cell(Problem.maxFE/Problem.N, 1);
            AllPopulationHistory  = cell(Problem.maxFE/Problem.N, 1);
            PredictedFrontHistory = {};
            
            AverageObjs = cell(length(Problem.maxFE/Problem.N), Problem.M);

			gen = 1;
            genPred = 1;
            
            % number of future generations to predict
			time_horizon = 2;

			% number of antecedent nondominated solutions used to train ML/TS model 
			neighbors = 60;
               
            % minimum of generations to be considered
%             min_gen = 20;
            
            % time between generation predictions
            window = 20;

			path = "~/Dev/platemo4-2/population_analysis/data_nsbidico_elm_de_pred_th" + string(time_horizon) + "_nh"+ string(neighbors) + "/";
			
			problem_dir = path + class(Problem) + "/";
			if ~exist(problem_dir, 'dir')
				mkdir(problem_dir);
			end

			file_exec = problem_dir + "exec.mat";
			if ~exist(file_exec, 'file')
				execution = 1;
				save(file_exec, 'execution');
			else
				load(file_exec, '-mat', 'execution');
				execution = execution + 1;
				save(file_exec, 'execution');
			end

			file_dir = problem_dir + string(execution) + "/";
			if ~exist(file_dir, 'dir')
				mkdir(file_dir);
			end

			pred = true;
			% Optimization	
			while Algorithm.NotTerminated(Population)		

				% [Cr, F, proM, disM] = Algorithm.ParameterSet(CrPool(randi([1, length(CrPool)], 1)), FPool(randi([1, length(CrPool)], 1)), 1, 20);

				AllPop     = [Population,ArcPop];
				MatingPool = MatingSelection(Population,ArcPop,Problem.N);   
								
				% a = randperm(Problem.N, Problem.N);
				% b = randperm(Problem.N, Problem.N);

				% indexes = find(a==b);
				
				% while ~isempty(indexes)
				% 	for i = indexes
				% 		[b(i)]=randperm(Problem.N, 1);
				% 	end										
				% 	indexes = find(a==b);					
				% end
							                 

                % ######################################################################
                % ######################################################################

                
                %if gen > (100 + window)
                %    neighbors = 100;
                %end

                
                if pred && gen > (neighbors + window) && (gen - genPred) >= window 
                    
                    
                    AllPopulationHistory{gen} = [Population, ArcPop];
                    NondominatedHistory{gen} = NondominatedHistory{gen-1};

                    PredictedFront.gen    = gen;
                    PredictedFront.time_horizon = time_horizon;
                    
                    metric  = {'euclidean', 'cityblock'};

                    Data    = IdentifyClosestAllPop(AllPopulationHistory, ...
                        NondominatedHistory, neighbors, gen, metric{1});

                    % prediction in the objective space
                    [PredictedOS, SampledOS] = OSP(Data, time_horizon, ...
                        neighbors, gen, Problem);

				    PredictedFront.front   = PredictedOS;
                    PredictedFront.sampled = SampledOS;
                    
                    % prediction in the decision space
                    [PredictedDS, SampledDS] = DSP(Data, time_horizon, ...
                        neighbors, gen, Problem, PredictedOS);
                    
                    % get solutions closest to the predicted objective
                    % space
                    Solutions = ClosestSolutions(PredictedOS, SampledDS);


                    PredictedVariables.variables = PredictedDS;
                    PredictedVariables.sampled   = SampledDS;
                    PredictedVariables.solutions = Solutions;
                    
                    Offspring = Solutions;

                    genPred = gen;

                    save(file_dir + string(gen) + "_pred.mat", ...
                        'PredictedFront', 'PredictedVariables');

                    fid = fopen(file_dir + "gens_pred.txt", 'a+');
                    fprintf(fid, "%d\n", gen);
                    fclose(fid);
                
                else
                    Offspring = OperatorDE(Problem, MatingPool(1:end), ...
                                    MatingPool(randi(Problem.N,1,Problem.N)), ...
                                    MatingPool(randi(Problem.N,1,Problem.N)), ...
                                    {Cr, F, proM, disM});
                end
                

                ArcPop     = UpdateArc([AllPop, Offspring], Problem.N);				
				[Population,~,Nondominated,~, nd_pred, sr] = EnvironmentalSelection([Population, Offspring], Problem.N, gen, genPred);
                
                AllPopulationHistory{gen} = [Population, ArcPop];

                NondominatedHistory{gen} = [Nondominated, ArcPop];
                NDSolutions = NondominatedHistory{gen};

                save(file_dir + "ndh_" + string(gen) + ".mat", 'NDSolutions');
                save(file_dir + "nd_pred" + string(gen) + ".mat", 'nd_pred');
                save(file_dir + "sr" + string(gen) + ".mat", 'sr');

                result = {gen, Population, ArcPop};
				save(file_dir + string(gen) + ".mat", 'result');
				gen = gen + 1;

			end
        end
	end
end
