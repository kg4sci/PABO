class USwitcher:
    def __init__(self, window_size=5, cooling_period=3):
        self.window_size = window_size        
        self.cooling_period = cooling_period  
        self.best_utility = -float('inf')     
        self.utility_window = []              
        self.current_mode = 'explore'         
        self.cooling_counter = 0              

    def update(self, new_utility):
       
        
        self.best_utility = max(self.best_utility, new_utility)
        
       
        self.utility_window.append(new_utility)
        if len(self.utility_window) > self.window_size:
            self.utility_window.pop(0)
        
       
        if self.cooling_counter > 0:
            self.cooling_counter -= 1
            return False
        
        
        if len(self.utility_window) < self.window_size:
            return False
        
        
        if all(u < self.best_utility for u in self.utility_window):
            self._trigger_switch()
            return True
        
        return False

    def _trigger_switch(self):
       
        
        self.current_mode = 'explore' if self.current_mode == 'exploit' else 'exploit'
        
       
        self.utility_window = []
        self.cooling_counter = self.cooling_period

    def get_mode(self):
        
        return self.current_mode