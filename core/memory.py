import torch

class MemoryBuffer:
    def __init__(self, capacity=200):
        self.capacity = capacity
        self.X, self.y = [], []
        
    def update(self, X_new, y_new):
        self.X.extend(X_new)
        self.y.extend(y_new)
        if len(self.X) > self.capacity:
            self.X = self.X[-self.capacity:]
            self.y = self.y[-self.capacity:]
            
    def get_batch(self):
        if len(self.X) == 0: 
            return None, None
        return torch.stack(self.X), torch.stack(self.y)
