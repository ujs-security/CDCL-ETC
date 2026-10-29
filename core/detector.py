import torch
import numpy as np

class ConceptDriftDetector:
    def __init__(self, W=10, T_base=30, omega=3, gamma=0.9, tau_0=0.1):
        self.W = W
        self.T_base = T_base
        self.omega = omega
        self.gamma = gamma
        self.tau = tau_0
        self.dc_seq = []
        self.stable_conf = []
        
    def update(self, disc_outputs, current_preds):
        avg_disc = torch.mean(disc_outputs).item()
        dc_t = abs(avg_disc - 0.5)
        self.dc_seq.append(dc_t)
        
        if len(self.dc_seq) >= self.W:
            smoothed_dc = np.mean(self.dc_seq[-self.W:])
            mean_conf = torch.mean(current_preds).item()
            
            if len(self.stable_conf) > 0:
                c_stable = np.mean(self.stable_conf)
            else:
                c_stable = 1.0
                
            if smoothed_dc > self.tau and mean_conf < self.gamma * c_stable:
                return True, smoothed_dc
                
            self.stable_conf.append(mean_conf)
            if len(self.stable_conf) > self.T_base:
                self.stable_conf.pop(0)
                
        return False, 0.0

    def recalibrate(self, new_stable_dcs):
        self.tau = np.mean(new_stable_dcs) + self.omega * np.std(new_stable_dcs)
        self.dc_seq = []