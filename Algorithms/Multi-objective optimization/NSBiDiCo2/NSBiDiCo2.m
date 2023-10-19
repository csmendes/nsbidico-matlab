classdef NSBiDiCo2 < ALGORITHM
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
			
			[Cr, F, proM, disM] = Algorithm.ParameterSet(0.99, 0.9 , 1, 20);
			
			NondominatedHistory   = cell(Problem.maxFE/Problem.N, 1);
            AllPopulationHistory  = cell(Problem.maxFE/Problem.N, 1);
            PredictedFrontHistory = {};

			gen = 1;
            startGen = Inf;
            counter = 0;
			% Generate random population
			Population = Problem.Initialization();
			ArcPop     = [];

			% number of future generations to predict
			lambda = 2;

			% number of antecedent nondominated solutions used in the predictor 
			neighbors = 50;           

			% Optimization	
			while Algorithm.NotTerminated(Population)		

				% [Cr, F, proM, disM] = Algorithm.ParameterSet(CrPool(randi([1, length(CrPool)], 1)), FPool(randi([1, length(CrPool)], 1)), 1, 20);

				AllPop     = [Population,ArcPop];
				MatingPool = MatingSelection(Population,ArcPop,Problem.N);   
				% Offspring  = OperatorDE(Problem, Population, Population(randi(Problem.N,1,Problem.N)), Population(randi(Problem.N,1,Problem.N)), {Cr, F, proM, disM});

				[FrontNo,MaxFNo] = NDSort(MatingPool.objs, MatingPool.cons,Problem.N);
				Bests = MatingPool(FrontNo==1);

			
				% DE/current-to-best/1
				% Offspring  = DE(Problem, MatingPool, Bests(randi(length(Bests),1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), {Cr, F, proM, disM});
				
				% DE/current-to-rand/1
				% Offspring  = DE3(Problem, MatingPool, MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), {Cr, F, proM, disM});

				% DE/rand/2
				% Offspring  = DE2(Problem, MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), {Cr, F, proM, disM});

				% DE/rand/1
				Offspring  = OperatorDE(Problem, MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), {Cr, F, proM, disM});
%
				% DE/rand-to-best/1
				% Offspring  = DE4(Problem, MatingPool(randi(Problem.N,1,Problem.N)), Bests(randi(length(Bests),1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), {Cr, F, proM, disM});

				% DE/best/2
				% Offspring  = DE2(Problem, Bests(randi(length(Bests),1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), MatingPool(randi(Problem.N,1,Problem.N)), {Cr, F, proM, disM});
				
				ArcPop     = UpdateArc([AllPop, Offspring], Problem.N);		

				Population = NSGAEnvironmentalSelection([Population, Offspring], Problem.N);		
				% [Population,FrontNo,CrowdDis] = NSGAEnvironmentalSelection([Population, Offspring], Problem.N);		
				
				
                % AllPopulationHistory{gen} = [Population, ArcPop];

				% [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,Problem.N);
				% NondominatedHistory{gen} = [Population(FrontNo == 1), ArcPop];
				
                % if ~isempty(NondominatedHistory{gen}) && size(NondominatedHistory{gen},2) >= Problem.N || counter > 1;                    
                %     counter = counter + 1;
                % end
    
				% if counter > 50
                %     PredictedFront.gen    = gen;
                %     PredictedFront.lambda = lambda;
				% 	PredictedFront.front  = OSP(AllPopulationHistory, NondominatedHistory, lambda, neighbors, gen, Problem.M);
                %     %save('path', 'PredictedFront');
                %     counter = 0;
                % end

				% gen = gen + 1;

			end
        end
	end
end

