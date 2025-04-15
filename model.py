#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
import torch.nn.functional as F

from typing import Union, Tuple, List


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
        drop_prob (float): Probability of zeroing an element by the dropout layer.
        pre_dropout (bool): Determines if the dropout layer is placed at the start (pre_dropout = True)
                            or at the end (pre_dropout = False) of the block.
    '''
    def __init__(self, 
                 out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]] = 3 , 
                 drop_prob: float = 0.35, 
                 pre_dropout: bool = False):
        super().__init__()
        
        layers = [nn.LazyConv2d(out_channels, kernel_size), nn.ReLU(),
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
    
    '''
    def __init__(self):
        super().__init__()
        self.cnn_body = nn.Sequential()
        
        self.cnn_body.add_module(
            'cnn_block_1',
            nn.Sequential(
                ConvBNDrop(64, kernel_size = 3, drop_prob = 0.35),
                ConvBNDrop(128, kernel_size = 3, drop_prob = 0.35),

                nn.LazyConv2d(256, kernel_size = 3), nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(kernel_size = 2)
            )
        )
        
        self.cnn_body.add_module(
            'cnn_block_2',
            nn.Sequential(
                ConvBNDrop(512, kernel_size = 3, drop_prob = 0.35, pre_dropout = True),
                ConvBNDrop(1024, kernel_size = 3, drop_prob = 0.35, pre_dropout = True),
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
            nn.LazyLinear(10)
        )
        
    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of base CNN.

        Args:
            X (Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            logits (Tensor): Output logits of shape (batch_size, 10), rows are the class scores.
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
    
    '''
    def __init__(self):
        super().__init__()
        self.subnet = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512), nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            DropConnectLinear(512, 512, drop_prob = 0.5), nn.ReLU(),
            nn.LazyLinear(10)
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the subnetwork.

        Args:
            X (Tensor): Input tensor of shape (batch_size, in_channels, height, width).
                        In a EnsNet model, this would be the divided feature maps from the base CNN.

        Returns:
            Tensor: Output logits of shape (batch_size, 10), rows are the class scores.
        '''
        return self.subnet(X)
        
class EnsNet(nn.Module):
    '''
    EnsNet model for classifying MNIST data, as described by https://arxiv.org/pdf/2003.08562v3.

    Args:
        num_subnets (int): Number of fully-connected subnetworks to include as voters.
                           This also determines the number of chunks to split the base CNN feature maps into.
    '''
    def __init__(self, num_subnets: int = 10):
        super().__init__()
        
        self.num_subnets = num_subnets
        
        # Create base CNN to extract features before division
        self.base_cnn = EnsNetBaseCNN()
        
        # Create fully connected subnetworks to get votes
        self.subnets = nn.ModuleList([
            EnsNetFCSN() for _ in range(num_subnets)
        ])
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Predicts the class label(s) (0-9) for the input image(s) 
        through majority vote among the base CNN and subnetworks.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            torch.Tensor: Class labels of each input sample in X.
                          Shape is (batch_size,), where each entry is an integer class label.
        '''
        # Get logits from the base CNN and subnets
        cnn_logits, subnet_logits, _ = self.forward(X)
        all_logits = [cnn_logits] + subnet_logits # num_voters = num_subnets + 1

        # Get predicted classes from the base CNN and each subnet
        pred_classes = torch.stack(all_logits, dim = 0).argmax(dim = -1) # Shape: (num_voters, batch_size)
 
        # Get majority vote among predicted classes
            # Note: torch.mode() is not implemented on MPS
        votes = [
            torch.bincount(vote_batch, minlength = 10).argmax()
            for vote_batch in pred_classes.transpose(1, 0)
        ]
            
        return torch.stack(votes)

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor],  torch.Tensor]:
        '''
        Forward pass of EnsNet model. 
        The base CNN produces a set of logits (class scores) as well as feature maps.
        The feature maps are then divided between several subnetworks 
        that each produce their own set of logits (class socores).

        Args:
            X (Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            cnn_logits (Tensor): Output logits from base CNN. 
                                 Shape is (batch_size, 10), rows are the class scores.
            subnet_logits (List[Tensor]): List of output logits (Tensors) from all subnetworks. 
                                          The i-th entry contains the output logits from the i-th subnetwork,
                                          and has shape (batch_size, 10).
            cnn_feat_maps (Tensor): Final feature maps from base CNN. 
                                    Shape is (batch_size, 2000, new_height, new_width)
        '''
        # Get logits and feature maps from base CNN
        cnn_logits, cnn_feat_maps = self.base_cnn(X)

        # Divide the feature maps into distinct chunks
            # If the number of channels isn't divisible by num_subnets, the last chunk may be smaller
        div_feat_maps = torch.chunk(cnn_feat_maps, chunks = self.num_subnets, dim = 1)

        # Get logits from subnetworks
        subnet_logits = []
        for feat_maps, subnet in zip(div_feat_maps, self.subnets):
            subnet_logits.append(subnet(feat_maps))
            
        return cnn_logits, subnet_logits, cnn_feat_maps