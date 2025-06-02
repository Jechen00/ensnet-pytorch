#####################################
# Imports & Dependencies
#####################################
import argparse
import torch
from torch import nn
from src import utils, data_setup, model, engine

# Setup random seeds
utils.set_seed(6)

# Setup hyperparameters
parser = argparse.ArgumentParser(fromfile_prefix_chars = '@')

parser.add_argument('-nw', '--num-workers', 
                    help = 'Number of workers for dataloaders.',
                    type = int, default = 0)
parser.add_argument('-ne', '--num-epochs', 
                    help = 'Number of epochs to train model for.', 
                    type = int, default = 15)
parser.add_argument('-bs', '--batch-size', 
                    help = 'Size of batches to divide training and testing set.',
                    type = int, default = 100)
parser.add_argument('-lr', '--learning-rate', 
                    help = 'Learning rate for the optimizers.', 
                    type = float, default = 0.001)
parser.add_argument('-p', '--patience', 
                    help = 'Number of epochs to wait, without improvement, before early stopping.', 
                    type = int, default = 5)
parser.add_argument('-md', '--min-delta', 
                    help = 'Minimum change in performance metric to reset early stopping counter.', 
                    type = float, default = 0.001)
parser.add_argument('-sd', '--save-dir', 
                    help = 'Directory to save the model. Required if `mod_name` is provided or `save_results` is True.',
                    type = str, default = None)
parser.add_argument('-mn', '--mod-name', 
                    help = "Filename for the saved model. Defaults to 'ensnet_model.pth' if save_dir is given.",
                    type = str, default = None)
parser.add_argument('-sr', '--save-results', 
                    help = 'If included, saves the returned results as .pkl files in `save_dir.`',
                    action = 'store_true')

args = parser.parse_args()


#####################################
# Training Code
#####################################
if __name__ == '__main__':
    print(f"{'#' * 50}\n"
          f'{utils.BOLD_START}Training hyperparameters:{utils.BOLD_END} \n'
          f'    - num_workers:   {args.num_workers} \n'
          f'    - num_epochs:    {args.num_epochs} \n'
          f'    - batch_size:    {args.batch_size} \n'
          f'    - learning_rate: {args.learning_rate} \n'
          f'    - patience:      {args.patience} \n'
          f'    - min_delta:     {args.min_delta} \n'
          f"{'#' * 50}") 

    # Get dataloaders
    train_dl, test_dl = data_setup.get_dataloaders(root = './mnist_data',
                                                   batch_size = args.batch_size,
                                                   num_workers = args.num_workers)

    # Get EnsNet model
    mod_kwargs = {
        'base_cnn': model.EnsNetBaseCNN,
        'subnet': model.EnsNetFCSN,
        'num_subnets': 10,
        'num_classes': len(train_dl.dataset.classes)
    }
    ensnet_mod = model.EnsNet(**mod_kwargs).to(utils.DEVICE)
    # ensnet_mod = torch.compile(ensnet_mod)
    
    # Get loss function and optimizers
    loss_fn = nn.CrossEntropyLoss()
    cnn_optimizer = torch.optim.Adam(params = ensnet_mod.base_cnn.parameters(), lr = args.learning_rate)
    
    subnets_optimizers = [torch.optim.Adam(params = subnet.parameters(), lr = args.learning_rate)
                          for subnet in ensnet_mod.subnets]
    
    # Train model
    cnn_res, subnets_res, ensemble_res = engine.train(ensnet = ensnet_mod,
                                                      train_dl = train_dl,
                                                      test_dl = test_dl,
                                                      loss_fn = loss_fn,
                                                      cnn_optimizer = cnn_optimizer,
                                                      subnets_optimizers = subnets_optimizers,
                                                      num_epochs = args.num_epochs,
                                                      patience = args.patience,
                                                      min_delta = args.min_delta,
                                                      device = utils.DEVICE,
                                                      save_dir = args.save_dir,
                                                      mod_name = args.mod_name,
                                                      save_results = args.save_results)
