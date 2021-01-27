import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.nn.init as init

class AE(nn.Module):

    """
    Class to generate an autoencoder based on the encoder dimensions
    Parameters:
        - dims : array representing the dimensions of the encoder
        - activation : activation function to use, 'sigmoid' or 'relu'
        - dropout : dropout rate in the encoder (from 0 to 1)
    """
    def __init__(self,dims,activation="relu",dropout=0.0,slope=0.0,batch_norm=False):
        super(AE,self).__init__()
        self.in_dim = dims[0]
        self.out_dim = dims[-1]
        self.n_layers = len(dims)-1
        self.dropout = dropout
        self.slope = slope
        self.batch_norm = batch_norm

        print("Batch norm : ",self.batch_norm)

        encoder_list = []
        decoder_list = []
        
        # Building the autoencoder
        
        # Encoder
        for i in range(1,len(dims)):
            encoder_list.append(nn.Linear(dims[i-1],dims[i]))
            if i < len(dims)-1 :
                if activation == "sigmoid" :
                    encoder_list.append(nn.Sigmoid())
                elif activation == "leakyrelu":
                    encoder_list.append(nn.LeakyReLU(negative_slope=self.slope))
                else:
                    encoder_list.append(nn.ReLU(inplace=True))
                if self.batch_norm :
                    encoder_list.append(nn.BatchNorm1d(dims[i]))
            if dropout > 0 and i < len(dims)-1:
                encoder_list.append(nn.Dropout(p=self.dropout))

        print(encoder_list)
        # Decoder
        for i in range(len(dims)-1,0,-1):
            decoder_list.append(nn.Linear(dims[i],dims[i-1]))
            if i > 1:
                if activation == "sigmoid" :
                    decoder_list.append(nn.Sigmoid())
                elif activation == "leakyrelu":
                    encoder_list.append(nn.LeakyReLU(negative_slope=self.slope))
                else:
                    decoder_list.append(nn.ReLU(inplace=True))
                if self.batch_norm :
                    decoder_list.append(nn.BatchNorm1d(dims[i-1]))

        self.encoder = nn.Sequential(*encoder_list)
        self.decoder = nn.Sequential(*decoder_list)


    def forward(self,x):
        x = x.view((-1,self.in_dim))
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self,x):
        x = x.view((-1,self.in_dim))
        x = self.encoder(x)
        return x
    
    def decode(self,x):
        x = x.view((-1,self.out_dim))
        x = self.decoder(x)
        return x
