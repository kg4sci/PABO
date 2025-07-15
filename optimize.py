
import numpy as np
from platypus import  Problem, Real
from NSGA.Wnsga import WeightedNSGAII 
from utility.chebyshev import chebyshev_utility
from model import GaussianProcess
from acquisitions import ei, compute_beta
import scipy
from benchmark import hatch
from select.USwitcher import USwitcher
from utility.zlcheb import IncrementalZengLearner


def optimize(w_true,initial_number, total_iterations,d,M,bound,bench_name,candidate_points,min_values,max_values):
       
       
        acquisition = ei  # Acquisition function
        batch_size = 1
        mode='explore' 
    
        preferences = []

      
        alpha =  [2] * M
        weights = np.random.dirichlet(alpha)
        learner = IncrementalZengLearner(M, memory_size=50)

        ###################### GP Initialization ###########################
        GPs = [GaussianProcess(d) for _ in range(M)]

        ###################### Evaluate Objective Functions ################
        def evaluation(xx):
          
            if bench_name=='hatch':
                return hatch(xx)
            
        indices = np.random.choice(candidate_points.shape[0], initial_number, replace=False)
        initial_candidates = candidate_points[indices]
        
        for i in range(initial_number):
            for j in range(M):
                obj=evaluation(initial_candidates[i])
                
                
                GPs[j].addSample(initial_candidates[i], obj[j])
        

       
        for i in range(M):
            GPs[i].fitModel()

        all_preferences = []
        batch_result = []  
        measured_points=[]
        utility_gaps = []  
        max_utility = -0.0
       
    
        uswitcher=USwitcher()

       ############################## Optimization Loop #############################
        for l in range(total_iterations):
            beta = compute_beta(l + 1, total_iterations)
        
            def MO(x):
                lambda_pref = -1
                x = np.asarray(x)
            
                acquisition_objectives = [
                    
                    acquisition(x, beta, GPs[i])[0] 
                    for i in range(M)
                ]
            
                return acquisition_objectives

      
        
            problem = Problem(d, M)
            for i in range(d):
              
              problem.types[i] = Real(bound[i][0], bound[i][1])
            problem.function = MO
            algorithm = WeightedNSGAII(problem, weights)
           
            algorithm.run(2000)
            cheap_pareto_set=[solution.variables for solution in algorithm.result]
            test_points=[]
            
            for i in range(len(cheap_pareto_set)):
                if (any((cheap_pareto_set[i] == x).all() for x in GPs[0].xValues))==False:
                 test_points.append(cheap_pareto_set[i])

        
         
           
            x_candidates=test_points

            objective_values = []


            for candidate in test_points:
                obj_values = evaluation(candidate)  
                objective_values.append(obj_values)
            

            uncertain_preferences = []
            all_predictions=[]
            
            for x in x_candidates:
           
                gp_predictions = [GPs[i].getPrediction(np.asarray(x))[0][0] for i in range(M)]

                all_predictions.append(gp_predictions)
                gp_predictions = (gp_predictions - min_values) / (max_values - min_values)
                chebyshev_value= chebyshev_utility([-x for x in gp_predictions], weights)
               
                
                uncertain_preferences.append([chebyshev_value,1])
           
            
            batch = []
            
            mode=uswitcher.get_mode()
            if mode=='explore':            
                UBs=[[GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]+beta*GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0] for i in range(M)] for x in x_candidates]
                LBs=[[GPs[i].getPrediction(np.asarray(np.asarray(x)))[0][0]-beta*GPs[i].getPrediction(np.asarray(np.asarray(x)))[1][0] for i in range(M)] for x in x_candidates]
                uncertaities= [scipy.spatial.Rectangle(UBs[i], LBs[i]).volume() for i in range(len(x_candidates))]
                indices = np.argsort(uncertaities)[::-1]
          
                batch.append(x_candidates[indices[batch_size-1]])
                measured_points.append(x_candidates[indices[batch_size-1]])
                batch_result.append(evaluation(x_candidates[indices[batch_size-1]]))
               
            else:
              
                chebyshev= [item[0] for item in uncertain_preferences]
                indices= [item[1] for item in uncertain_preferences]
                indice = np.argsort(chebyshev)[::-1] 
                
                
                batch.append(x_candidates[indice[0]])
                measured_points.append(x_candidates[indice[0]])
            
                
            
            
            
            best_pc_pair = None
            best_mi = -np.inf

            # Calculate mutual information for each pair of points
            for i, f1 in enumerate(all_predictions):
                for j, f2 in enumerate(all_predictions):
                   
                    if i >= j :
                      
                        continue
                   
                    mi,fn1,fn2 =calculate_mutual_information(
                        f1,f2, weights)
                   
                    if mi > best_mi:
                        best_mi = mi
                       
                        best_pc_pair = ([-x for x in fn1],[ -x for x in fn2])

            
            preferences=generate_pairwise_preferences(best_pc_pair, w_true)
            all_preferences.extend(preferences)
        
         
            weights,_ = learner.update_weights(preferences)
            
                
        
            batch_utilities = []
            for x_best in batch:
                objectives = evaluation(x_best)
                objective_values_norm=objectives
                objective_values_norm = (objectives - min_values) / (max_values - min_values)
                utility= chebyshev_utility( [-x for x in objective_values_norm], w_true)
                batch_utilities.append(utility)
                for i in range(M):
                    GPs[i].addSample(np.asarray(x_best), objectives[i])
                    GPs[i].fitModel()
              
            
            current_max_utility = max(batch_utilities)
            
            gap = max_utility - current_max_utility
            if len(utility_gaps) < 1 or gap < min(utility_gaps)  :
                    utility_gaps.append(gap)
                   
            else:
                    utility_gaps.append(min(utility_gaps))
                  
            uswitcher.update(current_max_utility)
         
         
       
        
        return utility_gaps,batch_result, measured_points


