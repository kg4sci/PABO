import numpy as np
from scipy.stats import norm, dirichlet
from scipy.optimize import minimize,differential_evolution, NonlinearConstraint

from scipy.special import expit 



def initialize_weights(num_objectives):
  
    w = np.ones(num_objectives) / num_objectives  
    return w


def phi(z):
   
    return 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * z ** 2)

def Phi(z):
   
    return norm.cdf(z)
def calculate_consistency(preferences):
    
    if not preferences:
        return 1.0  
    
    
    consistent_count = 0
    for f1, f2, pref in preferences:
        if (f1 > f2 and pref == 1) or (f1 < f2 and pref == 0):
            consistent_count += 1
    return consistent_count / len(preferences)


    

def chebyshev_utility(f_values, weights):
   
 
    normalized_values = f_values / weights
    
   
    chebyshev_part = np.min(normalized_values)
    
    return chebyshev_part

def log_posterior(w, pc_preferences):
   
    log_likelihood = 0
    w= w / np.sum(w)
    for f1, f2 in pc_preferences:
        
        delta_u = chebyshev_utility(f1, w) - chebyshev_utility(f2, w)
        log_likelihood += np.log(np.clip(Phi(delta_u), 1e-10, 1))
    
    prior_alpha= [2] * len(w)
    
   
    log_prior = dirichlet.logpdf(w, prior_alpha)
    
    return log_likelihood + log_prior


def update_weights(preferences, num_objectives, current_weights):
    
   
    current_weights = np.array(current_weights) / np.sum(current_weights)
   
    result = minimize(
        fun=lambda w: -log_posterior(w, preferences),
        x0=current_weights,  
        bounds=[(1e-6, 1.0) for _ in range(num_objectives)],
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  
        method='SLSQP' ,
        options={'maxiter': 1000, 'ftol': 1e-4, 'eps': 1e-6}  
    )
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)



def interact_with_decision_maker( paired_samples,weights):
            
       
    
       
        normalized1=paired_samples[0]
        normalized2=paired_samples[1]
        print(f"Pair :")
        # Compute utilities
        utility1 = chebyshev_utility(normalized1,weights)
        utility2 = chebyshev_utility(normalized2,weights)
            # For Option 1, show sample1 along with its corresponding target value
        print(f"Option 1: Target: {paired_samples[0]},utility: {utility1}")
            
            # For Option 2, show sample2 along with its corresponding target value
        print(f"Option 2: Target: {paired_samples[1]},utility: {utility2}")
            
            # Ask the decision-maker to choose their preferred option
        preferred = int(input("Which option do you prefer? (1 for Option 1, 2 for Option 2): "))
            
     

      
     
        if preferred == 1:
            return [(normalized1, normalized2)]
        else:
            return [(normalized2, normalized1)]
      
        
    


