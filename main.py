import torch
from sklearn.metrics import f1_score
from models.backbone import MultiChannelBackbone
from core.detector import ConceptDriftDetector
from core.continual import ContinualLearner
from core.memory import MemoryBuffer
from utils.data_loader import get_stream_dataloader

def run_experiment(dataset_name):
    stream_loader = get_stream_dataloader(dataset_name, window_size=50)
    model = MultiChannelBackbone(num_classes=2)
    memory = MemoryBuffer(capacity=200)
    learner = ContinualLearner(model, memory)
    detector = ConceptDriftDetector()
    
    y_true_all, y_pred_all = [], []
    
    for w_idx, (X_win, y_win) in enumerate(stream_loader):
        model.eval()
        with torch.no_grad():
            logits, features = model(X_win)
            preds = torch.argmax(logits, dim=1)
            confs = torch.softmax(logits, dim=1).max(dim=1)[0]
            
        y_true_all.extend(y_win.numpy())
        y_pred_all.extend(preds.numpy())
        
        disc_outputs = torch.rand(X_win.shape[0]) 
        is_drift, smoothed_dc = detector.update(disc_outputs, confs)
        
        if is_drift:
            learner.update(X_win, y_win, smoothed_dc, detector.tau)
            detector.recalibrate([0.1, 0.15, 0.12]) 

    final_f1 = f1_score(y_true_all, y_pred_all, average='macro')
    print(f"{dataset_name} | Final Macro-F1: {final_f1 * 100:.2f}%")

if __name__ == "__main__":
    run_experiment("ISCXVPN")
