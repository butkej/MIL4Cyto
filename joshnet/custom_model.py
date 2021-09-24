'''
This is my own pytorch implementation of an attention-based Deep Learning Model suited for multiple instance learning.

Main inspiration from Ilse et al. 2018 - Attention-based Deep Multiple Instance Learning

https://github.com/AMLab-Amsterdam/AttentionDeepMIL (Original implementation in PyTorch)

Heavily modified and expanded upon for use in CARS/SRS based Whole Slide Imaging of Urine Bladder Cancer Cell Identification/Classification
'''

# IMPORTS
#########

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# project specific imports (repo-internal)
from joshnet import mil_metrics

# UTILS
#######

def save_checkpoint(state, is_best, savepath):
    ''' Save model and state stuff if a new best is achieved
    Used in fit function in main.
    '''
    if is_best:
        print('--> Saving new best model')
        torch.save(state, savepath)

def load_checkpoint(loadpath, model, optim):
    ''' loads the model and its optimizer states from disk.
    '''
    checkpoint = torch.load(loadpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optim, epoch, loss

# MODEL
#######
class DeepAttentionMIL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear_nodes = 512
        self.attention_nodes = 128
        self.num_classes = 1 #3
        self.lam = 2 # lambda hyperparameter for adaptive weighting

        self.input_dim = (1,150,150) # for HE image tiles of Urine Sediment Cells

        self.use_bias = args.use_bias
        self.use_gated = args.use_gated # bool argument to use the gated attention mechanism
        self.use_adaptive = args.use_adaptive # bool argument to use adaptive weighting

        self.feature_extractor_0 = nn.Sequential(
                nn.Conv2d(in_channels=self.input_dim[0], out_channels=16, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),

        ).to('cuda:0')

        self.feature_extractor_1 = nn.Sequential(

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),

        ).to('cuda:1')

        self.feature_extractor_2 = nn.Sequential(

                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(2,2), bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2)

        ).to('cuda:2')

        size_after_conv = self._get_conv_output(self.input_dim)

        self.feature_extractor_3 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=size_after_conv, out_features=self.linear_nodes, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.5),

                nn.Linear(in_features=self.linear_nodes, out_features=self.linear_nodes, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.5)
                ).to('cuda:3') # bag of embeddings
        
        if not self.use_gated:
            self.attention = nn.Sequential(
                nn.Linear(self.linear_nodes, self.attention_nodes, bias=self.use_bias),
                nn.Tanh(),
                nn.Linear(self.attention_nodes, 1)
            ).to('cuda:3') # two-layer NN that replaces the permutation invariant pooling operator( max or mean normally, which are pre-defined and non trainable) with an adaptive weighting attention mechanism

        elif self.use_gated:
            self.attention_V = nn.Sequential(
                    nn.Linear(self.linear_nodes, self.attention_nodes),
                    nn.Tanh()
            ).to('cuda:3')

            self.attention_U = nn.Sequential(
                    nn.Linear(self.linear_nodes, self.attention_nodes),
                    nn.Sigmoid()
            ).to('cuda:3')

            self.attention = nn.Linear(self.attention_nodes, 1).to('cuda:3')

        self.classifier = nn.Sequential(
            nn.Linear(self.linear_nodes , self.num_classes),
            nn.Sigmoid()
        ).to('cuda:3')

    def forward(self, x):
        ''' Forward NN pass, declaring the exact interplay of model components
        '''
        x = x.squeeze(0) # compresses unnecessary dimensions eg. (1,batch,channel,x,y) -> (batch,channel,x,y)
        # transformation f_psi of instances in a bag
        hidden = self.feature_extractor_0(x)
        hidden = self.feature_extractor_1(hidden.to('cuda:1'))
        hidden = self.feature_extractor_2(hidden.to('cuda:2'))
        hidden = self.feature_extractor_3(hidden.to('cuda:3')) # N x linear_nodes
        
        # transformation sigma: attention-based MIL pooling
        if not self.use_gated:
            attention = self.attention(hidden) # N x num_classes

        elif self.use_gated:
            attention_V = self.attention_V(hidden)
            attention_U = self.attention_U(hidden)
            attention = self.attention(attention_V * attention_U)
            
        attention = torch.transpose(attention, 1, 0) # num_classes x N
        attention = F.softmax(attention, dim=1) #softmax over all N

        if not self.use_adaptive:
            z = torch.mm(attention, hidden) # num_classes x linear_nodes

        elif self.use_adaptive:
            # instance-level adaptive weighing attention [Li et al. MICCAI 2019]
            mean_attention = torch.mean(attention)
            thresh = nn.Threshold(mean_attention.item(), 0) # set elements in the attention vector to zero if they are <= mean_attention of the cycle
            positive_attention = thresh(attention.squeeze(0)) # vector of [1,n] to [n] and then threshold
            pseudo_positive = torch.where(positive_attention>0, torch.transpose(hidden,1,0), torch.tensor([0.],device='cuda:3')) # select all elements of the hidden feature embeddings that have sufficient attention
            positive_attention = positive_attention.unsqueeze(0) # reverse vector [n] to [1,n]

            negative_attention = torch.where(attention.squeeze(0) <= mean_attention, attention.squeeze(), torch.tensor([0.],device='cuda:3')) # attention vector with zeros if elements > mean_attention
            pseudo_negative = torch.where(negative_attention>0, torch.transpose(hidden,1,0), torch.tensor([0.],device='cuda:3')) # select all elements of the hidden feature embeddings matching this new vector
            negative_attention = negative_attention.unsqueeze(0)

            x_mul_positive = torch.mm(positive_attention, torch.transpose(pseudo_positive,1,0)) # pseudo positive instances N-N_in Matrix Mult.
            x_mul_negative = self.lam * torch.mm(negative_attention, torch.transpose(pseudo_negative,1,0)) # pseudo negative instances N_in Matrix Mult modfied by lambda hyperparameter (increases weightdifferences between pos/neg)
            z = x_mul_positive + x_mul_negative # see formula 2 of Li et al. MICCAI 2019
            
        # transformation g_phi of pooled instance embeddings 
        y_hat = self.classifier(z) 
        y_hat_binarized = torch.ge(y_hat, 0.5).float()
        return y_hat, y_hat_binarized, attention

    def _get_conv_output(self, shape):
        ''' generate a single fictional input sample and do a forward pass over 
        Conv layers to compute the input shape for the Flatten -> Linear layers input size
        '''
        bs = 1
        test_input = torch.autograd.Variable(torch.rand(bs, *shape)).to('cuda:0')
        output_features = self.feature_extractor_0(test_input)
        output_features = self.feature_extractor_1(output_features.to('cuda:1'))
        output_features = self.feature_extractor_2(output_features.to('cuda:2'))
        n_size = int(output_features.data.view(bs, -1).size(1))
        del test_input, output_features
        return n_size

    # COMPUTATION METHODS
    def compute_loss(self, X, y):
        ''' otherwise known as loss_fn
        Takes a data input of X,y (batches or bags) computes a forward pass and the resulting error.
        '''
        y = y.float()

        y_hat, y_hat_binarized, attention = self.forward(X)
        y_prob = torch.clamp(y_hat, min=1e-5, max=1. - 1e-5)

        loss_func = nn.BCELoss()
        loss = loss_func(y_hat, y)
        return loss, attention

    def compute_accuracy(self, X, y):
        ''' compute accuracy
        '''
        y = y.float()
        y = y.unsqueeze(dim=0)
        
        y_hat, y_hat_binarized, _  = self.forward(X)
        y_hat = y_hat.squeeze(dim=0)

        acc = mil_metrics.binary_accuracy(y_hat, y)
        return acc

###

class BaselineMIL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear_nodes = 512
        self.num_classes = 1 #3

        self.input_dim = (1,150,150)

        self.use_bias = args.use_bias
        self.use_max = args.use_max

        self.feature_extractor_0 = nn.Sequential(
                nn.Conv2d(in_channels=self.input_dim[0], out_channels=16, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),

        ).to('cuda:0')

        self.feature_extractor_1 = nn.Sequential(

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),


        ).to('cuda:1')

        self.feature_extractor_2 = nn.Sequential(

                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=(2,2), bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.MaxPool2d(2, stride=2)

        ).to('cuda:2')

        size_after_conv = self._get_conv_output(self.input_dim)

        self.feature_extractor_3 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=size_after_conv, out_features=self.linear_nodes, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.5),

                
                nn.Linear(in_features=self.linear_nodes, out_features=self.linear_nodes, bias=self.use_bias),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.5)
        ).to('cuda:3') # bag of embeddings
        

        self.classifier = nn.Sequential(
            nn.Linear(1 , self.num_classes),
            nn.Sigmoid()
        ).to('cuda:3')

    def forward(self, x):
        ''' Forward NN pass, declaring the exact interplay of model components
        '''
        x = x.squeeze(0) # compresses unnecessary dimensions eg. (1,batch,channel,x,y) -> (batch,channel,x,y)
        hidden = self.feature_extractor_0(x)
        hidden = self.feature_extractor_1(hidden.to('cuda:1'))
        hidden = self.feature_extractor_2(hidden.to('cuda:2'))
        hidden = self.feature_extractor_3(hidden.to('cuda:3')) # N x linear_nodes

        if not self.use_max:
            pooled = torch.mean(hidden, dim=[0,1], keepdim=True) # N x num_classes

        elif self.use_max:
            pooled = torch.max(hidden) # N x num_classes
            pooled = pooled.unsqueeze(dim=0)
            pooled = pooled.unsqueeze(dim=0)
            
        attention = torch.tensor([[0.5]])
        y_hat = self.classifier(pooled) 
        y_hat_binarized = torch.ge(y_hat, 0.5).float()
        return y_hat, y_hat_binarized, attention

    def _get_conv_output(self, shape):
        ''' generate a single fictional input sample and do a forward pass over 
        Conv layers to compute the input shape for the Flatten -> Linear layers input size
        '''
        bs = 1
        test_input = torch.autograd.Variable(torch.rand(bs, *shape)).to('cuda:0')
        output_features = self.feature_extractor_0(test_input)
        output_features = self.feature_extractor_1(output_features.to('cuda:1'))
        output_features = self.feature_extractor_2(output_features.to('cuda:2'))
        n_size = int(output_features.data.view(bs, -1).size(1))
        del test_input, output_features
        return n_size

    # COMPUTATION METHODS
    def compute_loss(self, X, y):
        ''' otherwise known as loss_fn
        Takes a data input of X,y (batches or bags) computes a forward pass and the resulting error.
        '''
        y = y.float()

        y_hat, y_hat_binarized, attention = self.forward(X)
        y_prob = torch.clamp(y_hat, min=1e-5, max=1. - 1e-5)

        loss_func = nn.BCELoss()
        loss = loss_func(y_hat, y)
        return loss, attention

    def compute_accuracy(self, X, y):
        ''' compute accuracy
        '''
        y = y.float()
        y = y.unsqueeze(dim=0)
        
        y_hat, y_hat_binarized, _  = self.forward(X)
        y_hat = y_hat.squeeze(dim=0)

        acc = mil_metrics.binary_accuracy(y_hat, y)
        return acc
