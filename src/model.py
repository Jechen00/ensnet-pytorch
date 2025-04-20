#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Tuple, List, Type


#####################################
# Model Classes
#####################################
class DropConnectLinear(nn.Linear):
    '''
    Implements a drop connection before a linear layer. 
    This is based on code from torchnlp: 
        https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/weight_drop.html.
    Note: The torchnlp implementation may not follow the original drop connect paper. 
        For one, it uses inverted dropout.
    Original Paper: https://proceedings.mlr.press/v28/wan13.html.
    
    Args:
        in_features (int): Number of features in the input.
        out_features (int): Number of features in the output.
        drop_prob (float): Probability of a weight being dropped during training.
    '''
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 drop_prob: float = 0.5, 
                 **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.drop_prob = drop_prob

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the drop connect layer.
        
        Args:
            X (Tensor): Input tensor of shape (batch_size, ..., in_features)
        Returns:
            Tensor: Output tensor of shape (batch_size, ..., out_features)
        '''
        # Drop random weights during training
        drop_weights = F.dropout(self.weight, p = self.drop_prob, training = self.training)
        
        return F.linear(X, drop_weights, self.bias)

class ConvBNDrop(nn.Module):
    '''
    A block for the base CNN of the EnsNet model. 
    It consists of a convolutional layer followed by batch normalization, 
    with dropout applied either before or after this.
    
    Args:
        out_channels (int): Number of channels in the output.
        kernel_size (int or Tuple[int, int]): Kernel size of the convolutional layer.
        padding (int, Tuple[int, int], or str): Padding of the convolutional layer.
        drop_prob (float): Probability of zeroing an element by the dropout layer.
        pre_dropout (bool): Determines if the dropout layer is placed at the start (pre_dropout = True)
                            or at the end (pre_dropout = False) of the block.
    '''
    def __init__(self, 
                 out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]] = 3 , 
                 padding: Union[int, Tuple[int, int], str] = 0,
                 drop_prob: float = 0.35, 
                 pre_dropout: bool = False):
        super().__init__()
        
        layers = [nn.LazyConv2d(out_channels, kernel_size, padding = padding), nn.ReLU(),
                  nn.BatchNorm2d(out_channels),
                  nn.Dropout(drop_prob)]
        
        if pre_dropout:
            # Move the last element (Dropout) to the front
            layers = [layers[-1]] + layers[:-1]
        
        self.conv_bn_drop = nn.Sequential(*layers)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the ConvBNDrop block.
        
        Args:
            X (Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, new_height, new_width)
        '''
        return self.conv_bn_drop(X)

class EnsNetBaseCNN(nn.Module):
    '''
    The base CNN to the EnsNet model used to classify MNIST data. This is used to get the final feature maps 
    as well as a single set of votes (logits) for the EnsNet model's majority vote prediction.
    The architecture follows from the EnsNet paper: https://arxiv.org/pdf/2003.08562v3.
    
    Note: In the paper, it's said that 'zero padding' is used for the first and last convolutional layers of each block.
          While this is fine, the padding of the middle convolutional layers don't have their padding specified.
          Moreover, to get the paper's base CNN output shape of 6x6 feature maps, 
          we would need a padding size that increases the output spatial dimensions of the middle convolutional layers 
          (i.e. padding > 1, with kernel_size = 3). With this in mind, I decided to set padding = 'same' for all
          convolutional layers except the last one. This keeps the same base CNN output shape of 6x6 feature maps, 
          while also ensuring that we aren't increasing the output spatial dimensions in each layer.
          
    Args:
        num_classes (int): Number of class labels.
    
    '''
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn_body = nn.Sequential()
        self.cnn_body.add_module(
            'cnn_block_1',
            nn.Sequential(
                ConvBNDrop(64, kernel_size = 3, padding = 'same', drop_prob = 0.35),
                ConvBNDrop(128, kernel_size = 3, padding = 'same', drop_prob = 0.35),

                nn.LazyConv2d(256, kernel_size = 3, padding = 'same'), nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(kernel_size = 2)
            )
        )
        self.cnn_body.add_module(
            'cnn_block_2',
            nn.Sequential(
                ConvBNDrop(512, kernel_size = 3, padding = 'same', drop_prob = 0.35, pre_dropout = True),
                ConvBNDrop(1024, kernel_size = 3, padding = 'same', drop_prob = 0.35, pre_dropout = True),
                ConvBNDrop(2000, kernel_size = 3, drop_prob = 0.35, pre_dropout = True),

                nn.MaxPool2d(kernel_size = 2),
                nn.Dropout(0.35)
            )  
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512), nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            DropConnectLinear(512, 512, drop_prob = 0.5), nn.ReLU(),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of base CNN.

        Args:
            X (Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            logits (Tensor): Output logits of shape (batch_size, num_classes), rows are the class scores.
            final_feature_maps (Tensor): Final feature maps before classifier. 
                                         Shape is (batch_size, 2000, new_height, new_width)
        '''
        final_feature_maps = self.cnn_body(X)
        logits = self.classifier(final_feature_maps)
        
        return logits, final_feature_maps

class EnsNetFCSN(nn.Module):
    '''
    A single fully connected subnetwork (FCSN) for the EnsNet model classifying MNIST data.
    The architecture follows from the EnsNet paper: https://arxiv.org/pdf/2003.08562v3.

    Args:
        num_classes (int): Number of class labels.
    
    '''
    def __init__(self, num_classes):
        super().__init__()
        self.subnet = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512), nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            DropConnectLinear(512, 512, drop_prob = 0.5), nn.ReLU(),
            nn.LazyLinear(num_classes)
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the subnetwork.

        Args:
            X (Tensor): Input tensor of shape (batch_size, in_channels, height, width).
                        In a EnsNet model, this would be the divided feature maps from the base CNN.

        Returns:
            Tensor: Output logits of shape (batch_size, num_classes), rows are the class scores.
        '''
        return self.subnet(X)
        
class EnsNet(nn.Module):
    '''
    EnsNet model consisting of a base CNN and multiple subnetworks. 
    The base CNN is responsible for producing both a set of logits and the final convolutional 
    feature maps. These feature maps are partitioned among the subnetworks, 
    each of which also produce their own set of logits. During inference, predictions are then made through majority vote.
    Reference: https://arxiv.org/pdf/2003.08562v3.

    Args:
        base_cnn (Type[nn.Module]): A class for the base CNN architecture.
                                    This must accept `num_classes` as an argument during instantiation.
                                    It's forward pass should return classification logits
                                    and the final feature maps before the classifier layers.
        subnet (Type[nn.Module]): A class for the architecture of each subnetwork.
                                  This must accept `num_classes` as an argument during instantiation.
                                  It's forward pass should return classification logits.
        num_subnets (int): Number of fully-connected subnetworks to include as ensemble voters. This also determines 
                           the number of chunks to split the base CNN feature maps into. Default is 10.

         num_classes (int): Number of class labels. For the MNIST dataset, this is set to 10 (default).
    '''
    def __init__(self, 
                 base_cnn: Type[nn.Module], 
                 subnet: Type[nn.Module],
                 num_subnets: int = 10, 
                 num_classes: int = 10):
        super().__init__()
        
        self.num_subnets, self.num_classes = num_subnets, num_classes
        
        # Create base CNN to extract features before division
        self.base_cnn = base_cnn(num_classes)
        
        # Create fully connected subnetworks to get votes
        self.subnets = nn.ModuleList([
            subnet(num_classes) for _ in range(num_subnets)
        ])
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Predicts the class labels, given a batch of input images.
        This is done through majority vote among the base CNN and subnetworks.

        Args:
            X (Tensor): Batch of input images. The shape is (batch_size, in_channels, height, width)
        Returns:
            Tensor: Class labels of each input sample in X.
                    Shape is (batch_size,), where each entry is an index for a class label.
        '''
        # Get logits from the base CNN and subnets
        cnn_logits, subnets_logits, _ = self.forward(X)

        # Get predictions
        return self.predict_with_logits(cnn_logits, subnets_logits)
    
    def predict_with_logits(self, 
                            cnn_logits: torch.Tensor, 
                            subnets_logits: List[torch.Tensor]) -> torch.Tensor:
        '''
        Predicts the class labels, given logits from the base CNN and subnetworks.
        This is done through majority vote among the base CNN and subnetworks.

        Args:
            cnn_logits (Tensor): Output logits from base CNN. 
                                 Shape is (batch_size, num_classes), rows are the class scores.
            subnets_logits (List[Tensor]): List of output logits (Tensors) from all subnetworks. 
                                           The i-th entry contains the output logits from the i-th subnetwork,
                                           and has shape (batch_size, num_classes).
        Returns:
            Tensor: Class labels of each input sample in X.
                    Shape is (batch_size,), where each entry is an index for a class label.
        '''
        for logit in subnets_logits:
            assert cnn_logits.shape == logit.shape, f'Shape mismatch: Shape of cnn_logits and tensors in subnets_logits must be the same.'

        all_logits = [cnn_logits] + subnets_logits # num_voters = num_subnets + 1

        # Get predicted classes from the base CNN and each subnet
        pred_classes = torch.stack(all_logits, dim = 0).argmax(dim = -1) # Shape: (num_voters, batch_size)
 
        # Get majority vote among predicted classes
            # Note: `torch.mode()` is not implemented on MPS, so I used `torch.bincount()` instead.
        votes = [
            torch.bincount(sample_votes, minlength = self.num_classes).argmax()
            for sample_votes in pred_classes.transpose(1, 0)
        ]
            
        return torch.stack(votes)
    
    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor],  torch.Tensor]:
        '''
        Forward pass of EnsNet model. 
        The base CNN produces a set of logits as well as feature maps.
        The feature maps are then partitioned (along channel dimension) among several subnetworks
        that each produce their own set of logits.

        Args:
            X (Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            cnn_logits (Tensor): Output logits from base CNN. 
                                 Shape is (batch_size, num_classes), rows are the class scores.
            subnets_logits (List[Tensor]): List of output logits (Tensors) from all subnetworks. 
                                           The i-th entry contains the output logits from the i-th subnetwork,
                                           and has shape (batch_size, num_classes).
            cnn_feat_maps (Tensor): Final feature maps from base CNN. 
                                    Shape is (batch_size, 2000, new_height, new_width)
        '''
        # Get logits and feature maps from base CNN
        cnn_logits, cnn_feat_maps = self.base_cnn(X)

        # Divide the feature maps into distinct chunks
            # If the number of channels isn't divisible by num_subnets, the last chunk may be smaller
        div_feat_maps = torch.chunk(cnn_feat_maps, chunks = self.num_subnets, dim = 1)

        # Get logits from subnetworks
        subnets_logits = []
        for feat_maps, subnet in zip(div_feat_maps, self.subnets):
            subnets_logits.append(subnet(feat_maps))
            
        return cnn_logits, subnets_logits, cnn_feat_maps