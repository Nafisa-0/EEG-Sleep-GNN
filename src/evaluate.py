import torch
from utils import compute_metrics

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)

        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

    return compute_metrics(y_true, y_pred)