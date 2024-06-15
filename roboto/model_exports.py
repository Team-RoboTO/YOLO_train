import torch
from ultralytics.engine.model import Model

def export_ultralytics_to_onnx(model:Model):
    if torch.cuda.is_available():
        model.export(format='onnx')
        
        
def export_ultralytics_to_tensorRT(model:Model):
    if torch.cuda.is_available():
        model.export(format='engine')