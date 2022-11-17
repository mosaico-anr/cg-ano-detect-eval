# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

import numpy as np
import torch
import torch.nn as nn
import logging, sys, time
from tqdm import trange

from ..utils.algorithm_utils import PyTorchUtils, AverageMeter

# Get logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class LSTM_Encoder(nn.Module):
    def __init__(self, n_features, lstm_dim, hidden_size):
        """The LSTM VAE Encoder.

        Args:
            n_features (int)    : The number of input features.
            lstm_dim (int)      : The LSTM hidden size.
            hidden_size (int)   : The hidden size.
        """
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size=n_features, hidden_size=lstm_dim)
        self.softplus = nn.Softplus()
        self.enc_linear_mean = nn.Linear(in_features=lstm_dim, out_features=hidden_size)
        self.enc_linear_sigma = nn.Linear(in_features=lstm_dim, out_features=hidden_size)
        
    def reparameterize(self, mu, log_var):
        """[summary]

        Args:
            mu      : The mean from the encoder's latent space.
            log_var : The log variance from the encoder's latent space.

        Returns:
                The tensor sampled from the random distribution.
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
        
    def forward(self, batch):
        """Forward function of the encoder.

        Args:
            batch : Batch input.

        Returns:
            The mean, the log variance and the sampled batch.
        """
        lstm_outputs, _ = self.encoder_lstm(batch)
        lstm_outputs = self.softplus(lstm_outputs)
        z_mean = self.enc_linear_mean(lstm_outputs)
        z_logvar = self.enc_linear_sigma(lstm_outputs)
        z = self.reparameterize(z_mean, z_logvar)
        return z_mean, z_logvar, z
    
class LSTM_Decoder(nn.Module):
    def __init__(self, n_features, lstm_dim, hidden_size):
        """The LSTM VAE Decoder.

        Args:
            n_features (int)    : The number of input features.
            lstm_dim (int)      : The LSTM hidden size.
            hidden_size (int)   : The hidden size.
        """
        super().__init__()
        self.decoder_lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_dim)
        self.softplus = nn.Softplus()
        self.dec_linear_mean = nn.Linear(in_features=lstm_dim, out_features=n_features)
        self.dec_linear_sigma = nn.Linear(in_features=lstm_dim, out_features=n_features)
        self.tanh = nn.Tanh()
        
    def forward(self, batch):
        """Forward function of the decoder.

        Args:
            batch : Batch input.

        Returns:
            The mean, the log variance.
        """
        lstm_outputs, _ = self.decoder_lstm(batch)
        lstm_outputs = self.softplus(lstm_outputs)
        z_mean = self.dec_linear_mean(lstm_outputs)
        z_logvar = self.dec_linear_sigma(lstm_outputs)
        z_logvar = self.tanh(z_logvar)
        return z_mean, z_logvar

class LSTM_VAE(nn.Module, PyTorchUtils):
    def __init__(self, n_features, hidden_size, lstm_dim, seed, gpu):
        """[summary]

        Args:
            n_features (int)    : The number of input features.
            hidden_size (int)   : The hidden size.
            lstm_dim (int)      : The LSTM hidden size.
            seed (int)          : The random generator seed.
            gpu (int)           : The number of the GPU device to use.
        """
        #super(LSTM_VAE, self).__init__()
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.encoder = LSTM_Encoder(n_features, lstm_dim, hidden_size)
        self.decoder = LSTM_Decoder(n_features, lstm_dim, hidden_size)
        
        
    def forward(self, batch):
        """Forward function of the decoder.

        Args:
            batch : Batch input.

        Returns:
            The mean, the log variance.
        """
        mu_z, logvar_z, z = self.encoder(batch)
        mu_x, sigma_x = self.decoder(z)
        var_z = torch.exp(logvar_z)
        #print(f'mu_x, sigma_x forward shape 1 = {mu_x.shape}, {sigma_x.shape}')
        
        # Compute Kullback-Leibler divergence
        kl_loss = -0.5*torch.sum(1 + logvar_z - var_z - torch.square(mu_z))
        #print(f'kl forward shape 1 = {kl_loss.shape}')
        #kl_loss = torch.mean(kl_loss)
        #print(f'kl forward shape 2 = {kl_loss.shape}')
        sigma_norm = sigma_x.abs()
        #eps = torch.finfo(sigma_norm.dtype).eps
        #sigma_norm = sigma_norm.clamp(min=eps)
        dist = torch.distributions.normal.Normal(loc=mu_x, scale=sigma_norm)
        log_px = -dist.log_prob(batch)
        #print(f'log_px forward shape 1 = {log_px.shape}')
        
        return mu_x, sigma_x, log_px, kl_loss
    
    def reconstruct_loss(self, x, mu_x, sigma_x):
        """The reconstruction loss

        Args:
            x       : The batch input.
            mu_x    : The mean coming from the decoder.
            sigma_x : The variance coming from the decoder.

        Returns:
                The reconstruction loss. 
        """
        var_x = torch.square(sigma_x)
        reconst_loss = -0.5 * torch.sum(torch.log(var_x)) + torch.sum(torch.square(x - mu_x) / var_x)
        # print(f'mu shape = {mu_x.shape}')
        # print(f'sigma shape = {sigma_x.shape}')
        # print(f'x shape = {x.shape}')
        # print(f'recons shape = {reconst_loss.shape}')
        #reconst_loss = reconst_loss.view(x.shape[0], -1)
        return reconst_loss #torch.mean(reconst_loss, dim=0)

    def mean_log_likelihood(self, log_px):
        """The mean log likelihood computation.

        Args:
            log_px : The batch log likelihood.

        Returns:
                The mean log likelihood.
        """
        
        #log_px = log_px.view(log_px.shape[0], -1)
        mean_log_px = torch.mean(log_px)
        return mean_log_px #.mean(dim=0)


def fit_lstm_vae_early_stopping(train_loader, val_loader, model, patience, num_epochs, lr,
                            writer, verbose=True):
    """The fitting function of the LSTM VAE.

    Args:
        train_loader (Dataloader)   : The train dataloader.
        val_loader (Dataloader)     : The val dataloader.
        model (nn.Module)           : The Pytorch model.
        patience (int)              : The number of epochs to wait for early stopping.
        num_epochs (int)            : The max number of epochs.
        lr (float)                  : The learning rate.
        writer (SummaryWriter)      : The Tensorboard Summary Writer.
        verbose (bool, optional)    : Defaults to True.

    Returns:
                        [nn.Module ]: The fitted model.
    """
    model.to(model.device)  # .double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    #train_loss_by_epoch = []
    #val_loss_by_epoch = []
    best_val_loss = np.inf
    best_val_epoch = 0
    epoch_wo_improv = 0
    best_params = model.state_dict()
    # assuming first batch is complete
    for epoch in trange(num_epochs):
        # If improvement continue training
        if epoch_wo_improv < patience:
            logging.debug(f'Epoch {epoch + 1}/{num_epochs}.')
            #if verbose:
                #GPUtil.showUtilization()
            # Train the model
            #logger.debug("Begin training...")
            train_loss, train_log_likehood, train_kl_loss, train_recons_loss = train_lstm_vae(train_loader, model, optimizer, epoch)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_loss, val_log_likehood, val_kl_loss, val_recons_loss = validation_lstm_vae(val_loader, model, optimizer, epoch)
            
            if verbose:
                logger.info(f"Epoch: [{epoch+1}/{num_epochs}] - Train loss: {train_loss:.2f} - Val loss: {val_loss:.2f}")
            
            # Write in TensorBoard
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_kl_loss', train_kl_loss, epoch)
            writer.add_scalar('train_recons_loss', train_recons_loss, epoch)
            writer.add_scalar('train_log_likelihood', train_log_likehood, epoch)
            
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_kl_loss', val_kl_loss, epoch)
            writer.add_scalar('val_recons_loss', val_recons_loss, epoch)
            writer.add_scalar('val_log_likehood', val_log_likehood, epoch)
            
            # Check if the validation loss improve or not
            if val_loss < best_val_loss :
                best_val_loss = val_loss
                epoch_wo_improv = 0
                best_val_epoch = epoch
                best_params = model.state_dict()
            elif val_loss >= best_val_loss:
                epoch_wo_improv += 1
            
        else:
            # No improvement => early stopping is applied and best model is kept
            model.load_state_dict(best_params)
            break
            
    return model


def train_lstm_vae(train_loader, model, optimizer, epoch, kullback_coef=0.28, reg_lambda=0.55):
    """The training step.

    Args:
        train_loader (Dataloader)       : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.
        kullback_coef (float)           : The Kullback-Leibler coefficient.
        reg_lambda (float)              : The regularization coefficient.

    Returns:
                The average loss on the epoch.
    """
    # Compute statistics
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    recons_loss_meter = AverageMeter()
    log_likehood_meter = AverageMeter()
    
    #
    end = time.time()
    model.train()
    train_loss = []
    for ts_batch in train_loader:
        ts_batch = ts_batch.float().to(model.device)
        mu_x, sigma_x, log_px, kl_loss = model(ts_batch)
        recons_loss = model.reconstruct_loss(ts_batch, mu_x, sigma_x)
        loss = recons_loss + kullback_coef*kl_loss
        
        mean_log_likehood = model.mean_log_likelihood(log_px)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # multiplying by length of batch to correct accounting for incomplete batches
        kl_loss_meter.update(kl_loss.item())
        recons_loss_meter.update(recons_loss.item())
        log_likehood_meter.update(mean_log_likehood.item())
        loss_meter.update(loss.item())
        #train_loss.append(loss.item()*len(ts_batch))

    #train_loss = np.mean(train_loss)/train_loader.batch_size
    #train_loss_by_epoch.append(loss_meter.avg)

    return loss_meter.avg, log_likehood_meter.avg, kl_loss_meter.avg, recons_loss_meter.avg
    
def validation_lstm_vae(val_loader, model, optimizer, epoch, kullback_coef=0.28):
    """The validation step.

    Args:
        val_loader (Dataloader)         : The val data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.
        kullback_coef (float)           : The Kullback-Leibler coefficient.

    Returns:
                The average loss on the epoch.
    """
    # Compute statistics
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    recons_loss_meter = AverageMeter()
    log_likehood_meter = AverageMeter()
    
    model.eval()
    #val_loss = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().to(model.device)
            
            mu_x, sigma_x, log_px, kl_loss = model(ts_batch)
            recons_loss = model.reconstruct_loss(ts_batch, mu_x, sigma_x)
            loss = recons_loss + kullback_coef*kl_loss
            mean_log_likehood = model.mean_log_likelihood(log_px)

            # multiplying by length of batch to correct accounting for incomplete batches
            kl_loss_meter.update(kl_loss.item())
            recons_loss_meter.update(recons_loss.item())
            log_likehood_meter.update(mean_log_likehood.item())
            loss_meter.update(loss.item())
            #train_loss.append(loss.item()*len(ts_batch))
            

        return loss_meter.avg, log_likehood_meter.avg, kl_loss_meter.avg, recons_loss_meter.avg

@torch.no_grad()
def predict_lstm_var_test_scores(model, test_loader):
    """The prediction step.

    Args:
        model (nn.Module)               : The PyTorch model.
        test_loader (Dataloader)        : The test dataloader.

    Returns:
                The reconstruction score 
    """
    model.eval()
    reconstr_scores = []
    latent_points = []
    outputs_array = []
    for ts_batch in test_loader:
        ts_batch = ts_batch.float().to(model.device)

        mu_x, sigma_x, log_px, kl_loss = model(ts_batch[:,-1])
        
        #recons_loss = model.reconstruct_loss(ts_batch, mu_x, sigma_x)
        #loss = recons_loss + kl_loss
        #print(log_px.shape)
        log_px = log_px.view(log_px.shape[0], -1)
        #print(log_px.shape)
        mean_log_likehood = torch.mean(log_px, dim=1)
        reconstr_scores.append(mean_log_likehood.cpu().numpy())
        
    reconstr_scores = np.concatenate(reconstr_scores)
    #outputs_array = np.concatenate(outputs_array)
    #print(reconstr_scores.shape)

    padding = np.zeros(len(ts_batch[0]) - 1)
    #print(padding.shape)
    #print(reconstr_scores.shape)

    reconstr_scores = np.concatenate([padding, reconstr_scores])

    return reconstr_scores