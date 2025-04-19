#####################################
# Imports, Dependencies, Constants
#####################################
import torch
import random
import numpy as np
import pickle
import os
from typing import Union, Dict, List

# Setup device and multiprocessing context
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    MP_CONTEXT = None
    PIN_MEM = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    MP_CONTEXT = 'forkserver'
    PIN_MEM = False
else:
    DEVICE = torch.device('cpu')
    MP_CONTEXT = None
    PIN_MEM = False

BOLD_START = '\033[1m'
BOLD_END = '\033[0m'


#####################################
# Functions
#####################################
def set_seed(seed: int = 0):
    '''
    Sets random seed and deterministic settings for reproducibility across:
        - PyTorch
        - NumPy
        - Python's random module
        - CUDA
    
    Args:
        seed (int): The seed value to set.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def save_model(model: torch.nn.Module,
               save_dir: str, 
               mod_name: str):
    '''
    Saves the `state_dict()` of a model to the directory 'save_dir.'

    Args:
        model (torch.nn.Module): PyTorch model that will be saved.
        save_dir (str): Directory to save the model to.
        mod_name (str): Filename for the saved model. 

    '''
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok = True)

    # Create save path
    save_path = os.path.join(save_dir, mod_name)

    # Save model's state dict
    torch.save(obj = model.state_dict(), f = save_path)

def save_results(results: Union[Dict, List],
                 save_dir: str,
                 save_name: str):
    '''
    Saves model results to the directory 'save_dir.'

    Args:
        results (Union[Dict, List]): A dictionary or list of model results to save.
        save_dir (str): Directory to save results to.
        save_name (str): Filename for the saved results. This needs to end with .pkl.

    '''
    assert save_name.endswith('.pkl'), 'save_name must end with .pkl.'

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok = True)

    # Create save path
    save_path = os.path.join(save_dir, save_name)    

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

def toggle_param_freeze(module: torch.nn.Module, freeze: bool):
    '''
    Helper function for freezing and unfreezing parameters of a module.

    Args:
        module (torch.nn.Module): PyTorch model whose parameters will be frozen/unfrozen.
        freeze (bool): If True, module parameters will be frozen (requires_grad = False).
                       If False, parameters will be unfrozen (requires_grad = True).
    '''
    for param in module.parameters():
        param.requires_grad = not freeze

def get_ensemble_accuracy(ensnet: torch.nn.Module,
                          dataloader: torch.utils.data.DataLoader,
                          device: torch.device):
    
    '''
    Calculates the accuracy of the full EnsNet over a given dataset.

    Args:
        ensnet (torch.nn.Module): EnsNet model to evaluate its accuracy.
        dataloader (torch.utils.data.DataLoader): Dataloader containing dataset to predict on and get accuracy.
        device (torch.device): Device to compute on.

    Returns:
        ensemble_acc (float): Accuracy of the EnsNet over the dataset.
    '''
    ensnet.eval() # Set model to evaluation mode
    
    ensemble_acc = torch.tensor(0.0, device = device) # Full EnsNet accuracy
    num_samps = len(dataloader.dataset)
    
    with torch.inference_mode():
        # Loop through dataloader
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            ensemble_acc += (ensnet.predict(X) == y).sum()
    
        # Average over samples
        ensemble_acc /= num_samps
        
    return ensemble_acc.item()


#####################################
# Classes
#####################################
class EarlyStopping():
    '''
    Early stopping class used to determine if training should be stopped once
    a given metric ('score') stops improving.

    Args:
        patience (int): Number of epochs to wait, with no improvement, before early stopping.
        min_delta (float): Minimum change in score to reset counter.
        mode (str): Either set to 'max' or 'min'. 
            - 'max' means the score should be maximized(e.g., accuracy or F1).
            - 'min' means the score should be minimized (e.g., validation loss).

    Reference: https://pytorch.org/ignite/_modules/ignite/handlers/early_stopping.html#EarlyStopping
    '''
    def __init__(self, patience: int, min_delta: float, mode):
        assert min_delta >= 0, 'min_delta should be positive.'
        assert mode in ('max', 'min'), "mode must be set to 'min' or 'max'."

        self.patience, self.min_delta, self.mode = patience, min_delta, mode
        self.counter = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.improvement = False
        self.stop = False

    def __call__(self, score):
        # If maximize, check for adequate increase in best score
        if (self.mode == 'max') and (score > self.best_score + self.min_delta):
            self.best_score = score
            self.improvement = True
            self.counter = 0 # Reset counter

            # If minimize, check for adequate decrease in best score
        elif (self.mode == 'min') and (score < self.best_score - self.min_delta):
            self.best_score = score
            self.improvement = True
            self.counter = 0 # Reset counter

        # No adequate change --> increase counter
        else:
            self.improvement = False
            self.counter += 1

            # If counter exeeds patience, early stopping is triggered
            if self.counter > self.patience:
                self.stop = True
                print(f'{BOLD_START}[ALERT]{BOLD_END} Early stopping triggered. Training has been stopped.')