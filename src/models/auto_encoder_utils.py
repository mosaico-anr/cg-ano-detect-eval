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


class AutoEncoderModel(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, sequence_length: int,
                 hidden_size: int, seed: int, gpu: int):
        """Auto-Encoder model architecture.

        Args:
            n_features (int)        : The number of input features.
            sequence_length (int)   : The window size length.
            hidden_size (int)       : The hidden size.
            seed (int)              : The random generator seed.
            gpu (int)               : The number of the GPU device.
        """
        # Each point is a flattened window and thus has as many features as sequence_length * features
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        input_length = n_features * sequence_length

        # creates powers of two between eight and the next smaller power from the input_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch, return_latent: bool=False):
        """Forward function of the Auto-Encoder.

        Args:
            ts_batch        : The batch input.
            return_latent   : If the latent vector must be returned. 
                              Defaults to False.

        Returns:
                The reconstructed batch.
        """
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (reconstructed_sequence, enc) if return_latent else reconstructed_sequence
    
    
def fit_with_early_stopping(train_loader, val_loader, model, patience, num_epochs, lr,
                            writer, verbose=True):
    """The fitting function of the Auto Encoder.

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
            train_loss = train(train_loader, model, optimizer, epoch)


            # Get Validation loss
            #logger.debug("Begin evaluation")
            val_loss = validation(val_loader, model, optimizer, epoch)
            
            if verbose:
                logger.info(f"Epoch: [{epoch+1}/{num_epochs}] - Train loss: {train_loss:.2f} - Val loss: {val_loss:.2f}")
            
            # Write in TensorBoard
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            
            # Check if the validation loss improve or not
            if val_loss < best_val_loss :
                best_val_loss = val_loss
                epoch_wo_improv = 0
                best_params = model.state_dict()
            elif val_loss >= best_val_loss:
                epoch_wo_improv += 1
            
        else:
            # No improvement => early stopping is applied and best model is kept
            model.load_state_dict(best_params)
            break
            
    # Reconstruct on validation data
    model.eval()
    val_reconstr_errors = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().to(model.device)
            output = model(ts_batch)[:, -1]
            error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
            val_reconstr_errors.append(error.cpu().numpy())
    if len(val_reconstr_errors) > 0:
        val_reconstr_errors = np.concatenate(val_reconstr_errors)
    return model, val_reconstr_errors


def train(train_loader, model, optimizer, epoch):
    """The training step.

    Args:
        train_loader (Dataloader)       : The train data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.

    Returns:
                The average loss on the epoch.
    """
    # Compute statistics
    loss_meter = AverageMeter()
    
    #
    model.train()
    for ts_batch in train_loader:
        ts_batch = ts_batch.float().to(model.device)
        output, latent = model(ts_batch, True)
        loss = nn.MSELoss(reduction="mean")(output, ts_batch)
        loss += 0.5*latent.norm(2, dim=1).mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        # multiplying by length of batch to correct accounting for incomplete batches
        loss_meter.update(loss.item())
        #train_loss.append(loss.item()*len(ts_batch))

    #train_loss = np.mean(train_loss)/train_loader.batch_size
    #train_loss_by_epoch.append(loss_meter.avg)

    return loss_meter.avg
    
def validation(val_loader, model, optimizer, epoch):
    """The validation step.

    Args:
        val_loader (Dataloader)         : The val data loader.
        model (nn.Module)               : The Pytorch model.
        optimizer (torch.optim)         : The Optimizer.
        epoch (int)                     : The max number of epochs.

    Returns:
                The average loss on the epoch.
    """

    # Compute statistics
    loss_meter = AverageMeter()
    
    model.eval()
    #val_loss = []
    with torch.no_grad():
        for ts_batch in val_loader:
            ts_batch = ts_batch.float().to(model.device)
            output, latent = model(ts_batch, True)
            loss = nn.MSELoss(reduction="mean")(output, ts_batch)
            loss += 0.5*latent.norm(2, dim=1).mean()
            #val_loss.append(loss.item()*len(ts_batch))
            loss_meter.update(loss.item())
        return loss_meter.avg
    
@torch.no_grad()
def predict_test_scores(model, test_loader, latent=False, return_output=False):
    """The prediction step.

    Args:
        model (nn.Module)               : The PyTorch model.
        test_loader (Dataloader)        : The test dataloader.
        latent (bool, optional)         : If latent variable is used. Defaults to False.
        return_output (bool, optional)  : If the reconstruction vector is returned. 
                                          Defaults to False.

    Returns:
                The reconstruction score 
    """
    model.eval()
    reconstr_scores = []
    latent_points = []
    outputs_array = []
    for ts_batch in test_loader:
        ts_batch = ts_batch.float().to(model.device)
        if latent:
            output, encoding = model(ts_batch, return_latent=latent)
            output = output[:, -1]
            latent_points.append(torch.squeeze(encoding).cpu().numpy())
        else:
            output = model(ts_batch)[:, -1]
        error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
        reconstr_scores.append(error.cpu().numpy())
        outputs_array.append(output.cpu().numpy())
    reconstr_scores = np.concatenate(reconstr_scores)
    outputs_array = np.concatenate(outputs_array)
    #print(reconstr_scores.shape)
    if latent:
        latent_points = np.concatenate(latent_points)
        padding = np.zeros((len(ts_batch[0]) - 1, latent_points.shape[1]))
        latent_points = np.concatenate([padding, latent_points])
    multivar = (len(reconstr_scores.shape) > 1)
    if multivar:
        padding = np.zeros((len(ts_batch[0]) - 1, reconstr_scores.shape[-1]))
    else:
        padding = np.zeros(len(ts_batch[0]) - 1)
    reconstr_scores = np.concatenate([padding, reconstr_scores])
    outputs_array = np.concatenate([padding, outputs_array])

    if latent and return_output:
        return_vars = (reconstr_scores, latent_points, outputs_array)
    elif latent:
        return_vars = (reconstr_scores, latent_points)
    elif return_output:
        return_vars = (reconstr_scores, outputs_array)
    else:
        return_vars = reconstr_scores

    return return_vars