import torch
import torch.nn as nn
from copy import deepcopy

class ContinualLearner:
    def __init__(self, model, memory, lambda_d=0.5, lambda_r=0.1, T_min=2.0, T_max=6.0):
        self.model = model
        self.memory = memory
        self.lambda_d = lambda_d
        self.lambda_r = lambda_r
        self.T_min = T_min
        self.T_max = T_max
        self.old_model = deepcopy(model)

    def update(self, X_cur, y_cur, smoothed_dc, tau):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        criterion_ce = nn.CrossEntropyLoss()
        
        rho_t = min(1, max(0, (smoothed_dc - tau) / tau))
        T_t = self.T_min + (self.T_max - self.T_min) * rho_t
        
        X_mem, y_mem = self.memory.get_batch()
        if X_mem is not None:
            X_train = torch.cat([X_cur, X_mem], dim=0)
            y_train = torch.cat([y_cur, y_mem], dim=0)
        else:
            X_train, y_train = X_cur, y_cur
            
        for epoch in range(5):
            optimizer.zero_grad()
            logits, _ = self.model(X_train)
            loss_ce = criterion_ce(logits, y_train)
            
            with torch.no_grad():
                old_logits, _ = self.old_model(X_train)
                p_old = torch.softmax(old_logits / T_t, dim=1)
                
            p_new = torch.log_softmax(logits / T_t, dim=1)
            loss_distill = -torch.mean(torch.sum(p_old * p_new, dim=1)) * (T_t**2)
            
            loss_reg = 0
            for p_new_param, p_old_param in zip(self.model.parameters(), self.old_model.parameters()):
                loss_reg += torch.sum((p_new_param - p_old_param)**2)
                
            loss = loss_ce + self.lambda_d * loss_distill + self.lambda_r * loss_reg
            loss.backward()
            optimizer.step()
            
        self.old_model = deepcopy(self.model)
        self.memory.update(X_cur, y_cur)