from platypus import NSGAII
import numpy as np
class WeightedNSGAII(NSGAII):
    def __init__(self, problem, weights, population_size=100):
        super().__init__(problem, population_size=population_size)
        self.weights = np.array(weights)  # 偏好权重向量
    
    def crowding_distance(self, solutions):
      
        for solution in solutions:
            solution.crowding_distance = 0.0
        
        num_objectives = len(solutions[0].objectives)
        for i in range(num_objectives):
           
            solutions.sort(key=lambda s: s.objectives[i])
            
           
            solutions[0].crowding_distance = float('inf')
            solutions[-1].crowding_distance = float('inf')
            
            min_obj = solutions[0].objectives[i]
            max_obj = solutions[-1].objectives[i]
            scale = max_obj - min_obj if max_obj != min_obj else 1.0
            
            for j in range(1, len(solutions)-1):
                solutions[j].crowding_distance += self.weights[i] * abs((solutions[j+1].objectives[i] - solutions[j-1].objectives[i]) )/ scale