import numpy as np
from scipy.stats import dirichlet, norm
from utility.chebyshev import chebyshev_utility
from scipy.stats import entropy


class IncrementalZengLearner:
    def __init__(self, M, init_weights=None, alpha_base=2.0,
                 sigma=0.5, min_weight=0.1, memory_size=30,
                 burnin_samples=1000, sample_steps=200):
        self.M = M
        self.weights = init_weights if init_weights is not None else np.ones(M)/M
        self.alpha = np.array([alpha_base]*M)
        self.sigma = sigma
    
        self.min_weight = 0
        self.memory = []
        self.memory_size = memory_size
        self.burnin = burnin_samples
        self.sample_steps = sample_steps
        self.step_counter = 0
        self.stab=True
       

    def _update_memory(self, new_pair):
       
        while isinstance(new_pair, list) and len(new_pair) == 1:
            new_pair = new_pair[0]
        
      
        if not (isinstance(new_pair, tuple) and len(new_pair) == 2):
            raise ValueError(f"Invalid input format: {new_pair}. Expected a tuple of two elements.")
        
       
        if len(self.memory) >= self.memory_size:
     
            self.memory.pop(0)
        self.memory.append(new_pair)

    def _proposal_distribution(self, current_weights):
      
        mix_ratio = 0.7 * np.exp(-self.step_counter / 1000)
        alpha = mix_ratio * current_weights * 10 + (1 - mix_ratio) * self.alpha
        return dirichlet(alpha)

    def _likelihood(self, pairs, w):
       
        likelihood = 1.0
        for f_i, f_i_prime in pairs:
            u_diff = (chebyshev_utility(f_i, w) - 
                     chebyshev_utility(f_i_prime, w))
            likelihood *= norm.cdf(u_diff / self.sigma)
        return likelihood
   

    def update_weights(self, new_pair):
       
        self._update_memory(new_pair)
        self.step_counter += 1
        
       
        effective_pairs = self.memory
        
      
        samples = [self.weights.copy()]
        current_weights = self.weights.copy()
        current_likelihood = self._likelihood(effective_pairs, current_weights)
        
        for _ in range(self.sample_steps):
           
            proposal = self._proposal_distribution(current_weights)
            w_proposal_raw = proposal.rvs()  
            w_proposal = np.maximum(w_proposal_raw.flatten(), self.min_weight)  
            w_proposal /= w_proposal.sum()
            
         
            prop_likelihood = self._likelihood(effective_pairs, w_proposal)
            
         
            prior_current = dirichlet.pdf(current_weights, self.alpha)
            prior_proposal = dirichlet.pdf(w_proposal, self.alpha)
            prior_ratio = prior_proposal / prior_current  
            
            ratio = (prop_likelihood / current_likelihood) * prior_ratio
            
            if np.random.rand() < np.minimum(1.0, ratio):
                current_weights = w_proposal
                current_likelihood = prop_likelihood
            
            samples.append(current_weights.copy())
        
       
        valid_samples = samples[-self.burnin:]
        self.weights = np.mean(valid_samples, axis=0)
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights /= self.weights.sum()
        
        return self.weights.copy(),valid_samples
    
