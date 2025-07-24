###model
#https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html

mport torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
#from torchmetrics.functional.regression import r2_score


def block(in_f, out_f, drop):
        return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.GELU(),
        nn.Dropout(drop)
        )

def out_block(in_f, out_f):
        return nn.Sequential(
        nn.Linear(in_f, out_f),
        #nn.Sigmoid() #output should be within [0,1]
        #nn.ReLU()
        )

class Encoder(nn.Module):
    def __init__(self, shape: list = [128, 64, 32], drop=.4):
        """
        Args:
           shape : Dim of Encoder Layers [input, hidden, ..., latent]
        """
        super().__init__()

        blocks = [block(in_f, out_f, drop) for in_f, out_f in zip(shape, shape[1:])]
        self.enc_blocks = nn.Sequential(*blocks)

    def forward(self, x):
            return self.enc_blocks(x)   
     
class Decoder(nn.Module):
    def __init__(self, shape: list = [32, 64, 128], drop=.4):
        """
        Args:
        shape : Dim of Decoder Layers [latent, hidden, ..., output]
        """
        super().__init__()

        blocks = [block(in_f, out_f, drop) for in_f, out_f in zip(shape[:-1], shape[1:-1])]
        blocks.append(out_block(shape[-2], shape[-1]))      # output should not be activated
        self.dec_blocks = nn.Sequential(*blocks)

    def forward(self, x):
            return self.dec_blocks(x)
    
class AutoEncoder(pl.LightningModule): 
    def __init__(self, shape: list = [[512, 256, 128, 64, 32, 16], [16, 32, 64, 128, 256, 512]], drop=.4, 
                 encoder_class: object = Encoder, decoder_class: object = Decoder) -> object:
        """
        Args:
        shape : Dim of AutoEncoder Layers [Encoder:[input, hidden, ..., latent], Decoder:[latent, hidden, ..., output]]
        drop : Dropout Rate
        """
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(shape[0], drop=drop)
        self.decoder = decoder_class(shape[1], drop=drop)

    def forward(self, x):
        """The forward function takes in the time series and returns the reconstructed time series."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_reconstruction_loss(self, batch):
        """Given a batch of time series, this function returns the reconstruction loss (MSE / R^2 in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x) #x[0]    [0] nur für predict
        loss = F.mse_loss(x_hat, x, reduction="sum") #mean
        #loss = r2_score(x_hat, x) #x[0]  [0] nur für predict
        return loss
    
    def _reconstruction(self, batch):
        """Given a batch of time series, this function returns the reconstruction loss (MSE / R^2 in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        
        return x_hat
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch))

    '''def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        _, label = batch
        self.log("test_loss", loss)
        self.log("label", label)'''
