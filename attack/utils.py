import torch
import sys
sys.path.append('.')
from models.model_factory import model_creator


def load_model(cfg,):
    model = model_creator(cfg)
    model = model.eval()
    return model
