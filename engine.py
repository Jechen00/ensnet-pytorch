#####################################
# Imports & Dependencies
#####################################
import torch

import numpy as np
from typing import Dict, Tuple, List, Sequence

import utils


#####################################
# Functions
#####################################
def cnn_train_step(ensnet: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   device: torch.device) -> Tuple[float, float]:
    '''
    Training step for the base CNN of an EnsNet model, with all subnetworks frozen.

    The full EnsNet training regime alternates between the base CNN and subnetworks.
    This step essentially just trains the base CNN independently, 
    since the loss is computed on the CNN logits and only the CNN parameters are updated.

    Args:
        ensnet (torch.nn.Module): EnsNet model that will be trained.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to train on.
        loss_fn (torch.nn.Module): Loss function used as the error metric.
        optimizer (torch.optim.Optimizer): Optimizater used to update base CNN parameters per batch.
        device (torch.device): Device to compute on.

    Returns:
        train_loss (float): The average loss of the base CNN over the training set.
        train_acc (float): The accuracy of the base CNN over the training set.
    '''

    ensnet.base_cnn.train() # Set base CNN to train mode
    ensnet.subnets.eval() # Set subnets to eval mode (Probably not needed b/c no subnets forward pass)
    
    # Unfreeze CNN and freeze subnetsworks
    utils.toggle_param_freeze(ensnet.base_cnn, freeze = False)
    utils.toggle_param_freeze(ensnet.subnets, freeze = True) # Probably not needed b/c no subnets forward pass
    
    # Track CNN train loss and accuracy
    train_loss = torch.tensor(0.0, device = device)
    train_acc  = torch.tensor(0.0, device = device)
    num_samps = len(dataloader.dataset)
    
    # Loop through batches
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad() # Clear old gradients
        
        with torch.autocast(device_type = device.type, dtype = torch.float16):
            y_logits, _ = ensnet.base_cnn(X) # Get CNN logits
            loss = loss_fn(y_logits, y) # Calculate loss
            
        train_loss += loss.detach() * X.shape[0]
        
        loss.backward() # Perform backpropagation
        optimizer.step() # Update parameters
        
        train_acc += (y_logits.argmax(dim = 1) == y).sum() # Calculate accuracy
        
    # Get average loss and accuracy
    train_loss = train_loss.item() / num_samps
    train_acc = train_acc.item() / num_samps

    return train_loss, train_acc

def subnets_train_step(ensnet: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       loss_fn: torch.nn.Module,
                       optimizers: Sequence[torch.optim.Optimizer],
                       device: torch.device) -> Tuple[List[float], List[float]]:
    '''
    Training step for the subnetworks of an EnsNet model, with the parameters of the base CNN frozen.

    The full EnsNet training regime alternates between the base CNN and subnetworks.
    As part of this process, each subnetwork uses a distinct subset of the base CNN's feature maps
    and is trained and updated independently of the others.

    Args:
        ensnet (torch.nn.Module): EnsNet model that will be trained.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to train on.
        loss_fn (torch.nn.Module): Loss function used as the error metric for each subnetwork.
        optimizers (Sequence[torch.optim.Optimizer]): A sequence of optimizers for the subnetworks.
                                                      The i-th entry should be the optimizer for the i-th subnetwork.
        device (torch.device) Device to compute on.

    Returns:
        train_loss (List[float]): List of the average losses for the subnetworks, calculated over the training set.
                                  The i-th entry contains the average loss of the i-th subnetwork.
        
        train_acc (List[float]): List of the accuracies for the subnetworks, calculated over the training set.
                                 The i-th entry contains the accuracy of the i-th subnetwork.
    '''

    assert len(optimizers) == ensnet.num_subnets, (
       'Length of subnets_optimizers need to match number of subnetsworks in EnsNet.'
    )
    
    ensnet.subnets.train() # Set subnets to train mode
    ensnet.base_cnn.eval() # Set base CNN to eval mode (used as feature extractor)
    
    # Unfreeze subnetsworks and freeze CNN
    utils.toggle_param_freeze(ensnet.subnets, freeze = False)
    utils.toggle_param_freeze(ensnet.base_cnn, freeze = True) # Probably not needed b/c torch.no_grad()
    
    # Track train loss and accuracy of each subnetswork
    train_loss = torch.zeros(ensnet.num_subnets, device = device) # Stores loss of each subnets
    train_acc = torch.zeros(ensnet.num_subnets, device = device) # Stores accuracy of each subnets
    num_samps = len(dataloader.dataset)
    
    # Loop through batches
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        batch_size = X.shape[0]
        
        # Get CNN feature maps without tracking gradients
        with torch.no_grad():
            _, cnn_feat_maps = ensnet.base_cnn(X)
        
            # Divide the feature maps into distinct chunks
            div_feat_maps = torch.chunk(cnn_feat_maps, chunks = ensnet.num_subnets, dim = 1)
            
        # Loop through subnets to independently train and update them
        for i, subnets in enumerate(ensnet.subnets):
            
            optimizers[i].zero_grad() # Clear old gradients
            
            with torch.autocast(device_type = device.type, dtype = torch.float16):
                y_logits = subnets(div_feat_maps[i]) # Get logits
                loss = loss_fn(y_logits, y) # Calculate loss
                
            train_loss[i] += loss.detach() * batch_size
        
            loss.backward() # Perform backpropagation
            optimizers[i].step() # Update parameters
        
            train_acc[i] += (y_logits.argmax(dim = 1) == y).sum() # Calculate accuracy

    # Get average loss and accuracy per sample for each subnets
    train_loss /= num_samps
    train_acc /= num_samps

    return train_loss.cpu().tolist(), train_acc.cpu().tolist()

def test_step(ensnet: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Dict:
    '''
    Testing step for an EnsNet model. This function independently computes the loss and accuracy for base CNN and each subnetwork
    in the ensemble, across the entire dataset in dataloader. 

    Args:
        ensnet (torch.nn.Module): EnsNet model that will be tested.
        dataloader (torch.utils.data.DataLoader): Dataloader containing data to test on.
        loss_fn (torch.nn.Module): Loss function used for both the base CNN and subnetworks.
        device (torch.device): Device to compute on.

    Returns:
        Dict: A dictionary containing the following:
            - 'cnn_loss' (float): Average loss of the base CNN over the dataset.
            - 'cnn_acc' (float): Accuracy of the base CNN over the dataset.
            - 'subnets_loss' (List[float]): List of average losses for each subnetwork. 
                                            The i-th entry corresponds to the i-th subnetwork.
            - 'subnets_acc' (List[float]): List of accuracies for each subnetwork.
                                           The i-th entry corresponds to the i-th subnetwork.
    '''

    ensnet.eval() # Set model to evaluation mode
    num_samps = len(dataloader.dataset)
    
    test_res = {
        'cnn_loss': torch.tensor(0.0, device = device),
        'cnn_acc': torch.tensor(0.0, device = device),
        'subnets_loss': torch.zeros(ensnet.num_subnets, device = device),
        'subnets_acc': torch.zeros(ensnet.num_subnets, device = device)
    }
    
    with torch.inference_mode():
        # Loop through batches
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            batch_size = X.shape[0]
            
            cnn_logits, subnets_logits, _ = ensnet(X) # Get logits
            
            # Calculate loss and accuracy for base CNN
            test_res['cnn_loss'] += loss_fn(cnn_logits, y) * batch_size
            test_res['cnn_acc'] += (cnn_logits.argmax(dim = 1) == y).sum()
            
            # Calculate loss and accuracy for each subnets
            for i, y_logits in enumerate(subnets_logits):
                test_res['subnets_loss'][i] += loss_fn(y_logits, y) * batch_size
                test_res['subnets_acc'][i] += (y_logits.argmax(dim = 1) == y).sum()
    
        # Get average loss and accuracy per sample
        test_res['cnn_loss'] = test_res['cnn_loss'].item() / num_samps
        test_res['cnn_acc'] = test_res['cnn_acc'].item() / num_samps
        test_res['subnets_loss'] = (test_res['subnets_loss'].cpu() / num_samps).tolist()
        test_res['subnets_acc'] = (test_res['subnets_acc'].cpu() / num_samps).tolist()
    
    return test_res

def train(ensnet: torch.nn.Module,
          train_dl: torch.utils.data.DataLoader,
          test_dl: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          cnn_optimizer: torch.optim.Optimizer,
          subnets_optimizers: Sequence[torch.optim.Optimizer],
          num_epochs: int,
          patience: int,
          min_delta: float,
          device: torch.device,
          save_mod: bool = True,
          save_dir: str = '',
          mod_name: str = ''):
    
    '''
    Trains an EnsNet model by alternating between the base CNN and the subnetworks.

    During each epoch, the base CNN training step (subnetworks frozen) is performed first,
    followed the the subnetwork training step (base CNN frozen). 
    Performance metrics (loss and/or accuracy) are computed on both the training and test sets 
    for the CNN, subnetworks, and the full ensemble. Early stopping is used to track improvements in ensemble test accuracy.
    Note that the training metrics of the CNN and subnetworks 
    are computed while their respective parameters are being updated.
    Conversely, the training metric(s) of the full ensemble 
    is computed at the end of the epoch, similar to the test step.

    Args:
        ensnet (torch.nn.Module): The EnsNet model containing a base CNN and multiple subnetworks.
        train_dl (torch.utils.data.DataLoader): Dataloader for training data.
        test_dl (torch.utils.data.DataLoader): Dataloader for testing data.
        loss_fn (torch.nn.Module): Loss function used during training and testing.
        cnn_optimizer (torch.optim.Optimizer): Optimizer for the base CNN.
        subnets_optimizers (Sequence[torch.optim.Optimizer]): Sequence of optimizers, where the i-th entry 
                                                              corresponds to the i-th subnetwork. 
                                                              Length must match the number of subnetworks.
        num_epochs (int): Number of epochs to train for.
        patience (int): Number of epochs to wait, with no improvement in accuracy, before early stopping.
        min_delta (float): Minimum change in accuracy to reset early stopping counter.
        device (torch.device): Device to compute on.
        save_mod (bool, optional): If True, saves the model after each epoch. Default is True.
        save_dir (str, optional): Directory to save the model to. Must be nonempty if save_mod is True.
        mod_name (str, optional): Filename for the saved model. Must be nonempty if save_mod is True.

    Returns:
        cnn_res (Dict): Dictionary containing training and test loss/accuracy for the base CNN.
        subnets_res (Dict): Dictionary containing training and test loss/accuracy for the subnetworks.
                            Each value is a list where the i-th entry is the loss/accuracy of the i-th subnetwork.
        ensemble_res (Dict): Dictionary containing ensemble accuracy over the training and test sets.
    '''

    if save_mod:
        assert save_dir, 'save_dir cannot be None or empty.'
        assert mod_name, 'mod_name cannot be None or empty.'
        
    assert len(subnets_optimizers) == ensnet.num_subnets, (
       'Length of subnets_optimizers need to match number of subnetsworks in EnsNet.'
    )
    
    # Stopper for early stopping based on ensemble test accuracy
    stopper = utils.EarlyStopping(patience = patience, 
                                  min_delta = min_delta, 
                                  mode = 'max')

    # Initialize results dictionary
    cnn_res = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': []}
    
    subnets_res = {'train_loss': [], 'train_acc': [],
                  'test_loss': [], 'test_acc': []}
    
    ensemble_res = {'train_acc': [], 'test_acc': []}
    
    for epoch in range(num_epochs):
        print(f"{utils.BOLD_START + '*'*25}\nEPOCH {epoch + 1}\n{'*'*25 + utils.BOLD_END}")
        
        # Train CNN and subnetworks using an alternating method
        print(f'{utils.BOLD_START}[UPDATE]{utils.BOLD_END} ' +
              'Training base CNN; parameters of subnetworks are frozen.')
        cnn_train_loss, cnn_train_acc = cnn_train_step(ensnet, train_dl, 
                                                       loss_fn, cnn_optimizer, device)

        print(f'{utils.BOLD_START}[UPDATE]{utils.BOLD_END} ' +
              'Training subnetworks; parameters of base CNN are frozen.')
        subnets_train_loss, subnets_train_acc = subnets_train_step(ensnet, train_dl,
                                                                   loss_fn, subnets_optimizers, device)
        
        
        # Perform test step for both CNN and subnetsworks
        test_res = test_step(ensnet, test_dl, loss_fn, device)
        
        # Store loss and accuracy values for CNN
        cnn_res['train_loss'].append(cnn_train_loss)
        cnn_res['train_acc'].append(cnn_train_acc)
        cnn_res['test_loss'].append(test_res['cnn_loss'])
        cnn_res['test_acc'].append(test_res['cnn_acc'])
        
        # Store loss and accuracy values for subnetsworks
        subnets_res['train_loss'].append(subnets_train_loss)
        subnets_res['train_acc'].append(subnets_train_acc)
        subnets_res['test_loss'].append(test_res['subnets_loss'])
        subnets_res['test_acc'].append(test_res['subnets_acc'])
        
        # Calculate average loss and accuracy over subnetworks (for logging)
        avg_subnets_train_loss = np.average(subnets_train_loss)
        avg_subnets_train_acc = np.average(subnets_train_acc)
        avg_subnets_test_loss = np.average(test_res['subnets_loss'])
        avg_subnets_test_acc = np.average(test_res['subnets_acc'])
        
        # Calculate and store end-of-epoch ensemble accuracies
            # This requires another pass of the data, might slow down training a bit
        ensemble_train_acc = utils.get_ensemble_accuracy(ensnet, train_dl, device)
        ensemble_test_acc = utils.get_ensemble_accuracy(ensnet, test_dl, device)
        ensemble_res['train_acc'].append(ensemble_train_acc)
        ensemble_res['test_acc'].append(ensemble_test_acc)
        
        # Print result summary for epoch
        print(f'{utils.BOLD_START}[CNN]{utils.BOLD_END:<12} | ' +
              f'train_loss = {cnn_train_loss:<10.4f} | ' +
              f'train_acc = {cnn_train_acc:<10.4f} | ' +
              f"test_loss = {test_res['cnn_loss']:<10.4f} | " +
              f"test_acc = {test_res['cnn_acc']:<10.4f}")
        
        print(f'{utils.BOLD_START}[SUBNETS]{utils.BOLD_END:<8} | ' +
              f'avg_train_loss = {avg_subnets_train_loss:<6.4f} | ' +
              f'avg_train_acc = {avg_subnets_train_acc:<6.4f} | ' +
              f'avg_test_loss = {avg_subnets_test_loss:<6.4f} | ' +
              f'avg_test_acc = {avg_subnets_test_acc:<6.4f}')
        
        print(f'{utils.BOLD_START}[FULL ENSNET]{utils.BOLD_END:} | ' +
              f"{'-'*23:<23} | " +
              f'train_acc = {ensemble_train_acc:<10.4f} | ' +
              f"{'-'*22:<22} | " +
              f'test_acc = {ensemble_test_acc:<10.4f}')
        
        # Determine if early stopping should be triggered
        stopper(ensemble_test_acc)
        if not stopper.stop:
            # Save model if there is improvement in test accuracy
            if stopper.improvement and save_mod:
                utils.save_model(ensnet, save_dir, mod_name)
                print(f'{utils.BOLD_START}[SAVED]{utils.BOLD_END} Adequate improvement in EnsNet test accuracy. Model saved.\n')
        else:
            break

    return cnn_res, subnets_res, ensemble_res