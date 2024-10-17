

class Boundary:
    
    @staticmethod
    def absorbing(x, x_prev, a, b):
        return min(max(x, a), b)
    
    # This is useless
    @staticmethod
    def invisibility(x, x_prev, a, b):
        return x
    
    @staticmethod
    def invisible_reflecting(x, x_prev, a, b):
        diff = x - x_prev
        
    
        
    
    