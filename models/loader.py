# models/loader.py
import torch
from .gimefive import GiMeFive
import os

def load_model(weights_path='weights/model.pth', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GiMeFive().to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weight file not found at {weights_path}")
    state = torch.load(weights_path, map_location=device)
    # if state saved as 'model_state' or raw state_dict, handle both
    if isinstance(state, dict) and 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state
    # remove possible "module." prefixes from DataParallel
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state[new_key] = v
    model.load_state_dict(new_state)
    model.eval()
    return model, device
