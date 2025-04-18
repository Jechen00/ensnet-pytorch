#####################################
# Imports & Dependencies
#####################################
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from . import utils
from typing import Tuple

# Transformations applied to each image
BASE_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(), # Convert to tensor and rescale pixel values to within [0, 1]
    transforms.Normalize(mean = [0.1307], std = [0.3081]) # Normalize with MNIST stats
])

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomAffine(degrees = 10, # Rotate up to -/+ 10 degrees
                            scale = (0.8, 1.2), # Scale between 80 and 120 percent
                            translate = (0.08, 0.08), # Translate up to -/+ 8 percent in both x and y
                            shear = 0.3),  # Shear up to -/+ 0.3 degrees
    transforms.ToTensor(), # Convert to tensor and rescale pixel values to within [0, 1]
    transforms.Normalize(mean = [0.1307], std = [0.3081]), # Normalize with MNIST stats
])


#####################################
# Functions
#####################################
def get_dataloaders(root: str, 
                    batch_size: int, 
                    num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    '''
    Creates training and testing dataloaders for the MNIST dataset.

    Args:
        root (str): Path to download MNIST data.
        batch_size (int): Size used to split training and testing datasets into batches.
        num_workers (int): Number of workers to use for multiprocessing. Default is 0.
    '''

    # Get training and testing MNIST data
    mnist_train = datasets.MNIST(root, download = True, train = True, 
                                transform = TRAIN_TRANSFORMS)
    mnist_test = datasets.MNIST(root, download = True, train = False, 
                                transform = BASE_TRANSFORMS)

    # Create dataloaders
    if num_workers > 0:
        mp_context = utils.MP_CONTEXT
        persistent_workers = True
    else:
        mp_context = None
        persistent_workers = False

    train_dl = DataLoader(
        dataset = mnist_train,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        multiprocessing_context = mp_context,
        pin_memory = utils.PIN_MEM,
        persistent_workers = persistent_workers
    )

    test_dl = DataLoader(
        dataset = mnist_test,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        multiprocessing_context = mp_context,
        pin_memory = utils.PIN_MEM,
        persistent_workers = persistent_workers
    )

    return train_dl, test_dl