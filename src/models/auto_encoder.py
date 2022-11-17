# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse, time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.tensorboard import SummaryWriter

from ..utils.algorithm_utils import Algorithm, PyTorchUtils, save_torch_algo
from ..utils.data_processing import get_sub_seqs, get_train_data_loaders, data_processing_naive, data_processing_random, get_data
from .auto_encoder_utils import AutoEncoderModel, fit_with_early_stopping, predict_test_scores
from ..utils.evaluation_utils import performance_evaluations



class AutoEncoder(Algorithm, PyTorchUtils):
    def __init__(self, name: str='auto_encoder', num_epochs: int=10, batch_size: int=128,
                 lr: float=1e-3, hidden_size: int=5, window_size: int=10,
                 train_val_percentage: float=0.25, verbose=True, seed: int=None,
                  gpu: int=None, patience: int=2, save_dir='data'):
        """Auto-Encoder algorithm for anomaly detection.

        Args:
            name (str, optional)                    : Algorithm's name. 
                                                      Defaults to 'auto_encoder'.
            num_epochs (int, optional)              : The number max of epochs. 
                                                      Defaults to 10.
            batch_size (int, optional)              : The batch size. Defaults to 128.
            lr (float, optional)                    : The optimizer learning rate. 
                                                      Defaults to 1e-3.
            hidden_size (int, optional)             : The AE hidden size. 
                                                      Defaults to 5.
            window_size (int, optional)             : The size of the moving window. 
                                                      Defaults to 10.
            train_val_percentage (float, optional)  : The ratio val/train. 
                                                      Defaults to 0.25.
            verbose (bool, optional)                : Defaults to True.
            seed (int, optional)                    : The random generator seed. 
                                                      Defaults to None.
            gpu (int, optional)                     : The number of the GPU device to use. 
                                                      Defaults to None.
            patience (int, optional)                : The number of epochs to wait for 
                                                      early stopping. Defaults to 2.
            save_dir (str, optional)                : The folder to save the outputs. 
                                                      Defaults to 'data'.
        """
        
        Algorithm.__init__(self, __name__, name, seed)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.torch_save = True
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        self.hidden_size = hidden_size
        self.window_size = window_size
        self.train_val_percentage = train_val_percentage
        self.model = None
        self.verbose = verbose
        # Get Tensorboard writer
        self.save_dir = save_dir #os.path.join(save_dir, f"{name}_opts")
        #self.writer = SummaryWriter(self.save_dir)
        self.init_params = {"name": name,
                            "num_epochs": num_epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "hidden_size": hidden_size,
                            "window_size": window_size,
                            "train_val_percentage": train_val_percentage,
                            "seed": seed,
                            "gpu": gpu,
                            "patience": patience,
                            "verbose" : verbose
                            }

        self.additional_params = dict()
        
    def anomaly_vector_construction(self, test_recons_errors, norm=2, sigma=3):
        """Apply the 3-sigma threshold to build anomaly vector prediction.

        Args:
            test_recons_errors (np.array): Reconstruction error vectors.
            norm (int, optional): The norm to used for normalization. Defaults to 2.
            sigma (int, optional): Sigma multiplicator coefficient. Defaults to 3.
            return_anomalies_score (bool, optional): If the normalized anomaly score must be returned.
                                                     Defaults to False.

        Returns:
            np.array: Anomaly vector.
        """
        # Check if mean error and std error has been compute
        try:
            train_std_err = self.additional_params['train_error_std']
            train_mean_err = self.additional_params['train_error_mean']
            
            test_norm = test_recons_errors - train_mean_err

            if norm == 1:
                test_norm = np.abs(np.mean(test_norm, axis=1))
            elif norm == 2:
                test_norm = np.sqrt(np.mean(test_norm**2, axis=1))
            else:
                print("Norm not implemented")

            # Create anomalies vectors
            anomalies = (test_norm <= sigma*train_std_err)
            anomalies = np.array(list(map(lambda val : 0 if val else 1, anomalies)))
            return anomalies, test_norm
        except ValueError:
            print("The model has not been trained ! No train error mean or std !")
            
        
    def compute_mean_std(self, train_data):
        """Compute the mean and std on training data after fitting.

        Args:
            train_data (pd.Dataframe): Training dataframe.
        """
        
        train_data = self.additional_params['processor'].transform(train_data)
        # Create the window on the data processed
        train_data_win = get_sub_seqs(train_data, window_size=self.window_size)
        # Compute train mean/std error on whole training data
        self.model.eval()
        whole_data_loader = DataLoader(dataset=train_data_win, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors = []
        with torch.no_grad():
            for ts_batch in whole_data_loader:
                ts_batch = ts_batch.float().to(self.model.device)
                output = self.model(ts_batch)[:, -1]
                error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
                reconstr_errors.append(error.cpu().numpy())
        if len(reconstr_errors) > 0:
            reconstr_errors = np.concatenate(reconstr_errors)
        self.additional_params['train_error_mean'] = np.mean(reconstr_errors, axis=0)
        self.additional_params['train_error_std'] = np.std(reconstr_errors, axis=None)
        

    def fit(self, train_data : pd.DataFrame, categorical_columns=None):
        """Fit the model.

        Args:
            train_data (pd.DataFrame): Training dataframe.
            categorical_columns (list, optional): Column to be one-hot encoded.
                                                Defaults to None.
        """
        # Select columns to keep
        all_columns = train_data.columns.tolist()
        if categorical_columns is not None:
            numerical_columns = [col for col in all_columns if col not in categorical_columns]
        else:
            numerical_columns = all_columns.copy()
       
        # Create the preprocessing steps
        numerical_processor = StandardScaler()
        if categorical_columns is not None:
            categorical_processor = OneHotEncoder(handle_unknown="ignore")
            processor = ColumnTransformer([
                ('one-hot', categorical_processor, categorical_columns),
                ('scaler', numerical_processor, numerical_columns)
            ])
        else :
            processor = ColumnTransformer([
                ('scaler', numerical_processor, numerical_columns)
            ])
        
        # Fit on the processor
        #print(train_data.shape)
        #print(categorical_columns, numerical_columns)
        train_data = processor.fit_transform(train_data)
        self.additional_params['processor'] = processor
        
                
        # Create the window on the data processed
        train_data_win = get_sub_seqs(train_data, window_size=self.window_size)

        train_loader, val_loader = get_train_data_loaders(train_data_win, batch_size=self.batch_size,
                                                                splits=[1 - self.train_val_percentage,
                                                                        self.train_val_percentage], seed=self.seed)
        
        self.model = AutoEncoderModel(train_data.shape[1], self.window_size, self.hidden_size, seed=self.seed, gpu=self.gpu)
        print(f"Fitting {self.name} model")
        writer = SummaryWriter(self.save_dir)
        self.model, _ = fit_with_early_stopping(train_loader, val_loader, self.model, patience=self.patience,
                                    num_epochs=self.num_epochs, lr=self.lr, writer=writer, verbose=self.verbose)
        print("Fitting done")
        
        # Compute train mean/std error on whole training data
        self.model.eval()
        whole_data_loader = DataLoader(dataset=train_data_win, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors = []
        with torch.no_grad():
            for ts_batch in whole_data_loader:
                ts_batch = ts_batch.float().to(self.model.device)
                output = self.model(ts_batch)[:, -1]
                error = nn.L1Loss(reduction='none')(output, ts_batch[:, -1])
                reconstr_errors.append(error.cpu().numpy())
        if len(reconstr_errors) > 0:
            reconstr_errors = np.concatenate(reconstr_errors)
        self.additional_params['train_error_mean'] = np.mean(reconstr_errors, axis=0)
        self.additional_params['train_error_std'] = np.std(reconstr_errors, axis=None)
        
    @torch.no_grad()
    def predict(self, test_data : pd.DataFrame, norm=2, sigma=3, if_shap=True, custom=False):
        """Predict on the test dataframe

        Args:
            test_data (pd.DataFrame): Test dataframe.
            norm (int, optional): The norm to used for normalization. Defaults to 2.
            sigma (int, optional): The sigma coefficient. Defaults to 3.
            if_shap (bool, optional): If Shap values is computed during prediction. Defaults to True.
            custom (bool, optional): If threshold optimization is performed. Defaults to False.

        Returns:
            np.array: Test predictions.
        """
        # Process the data
        test_data = self.additional_params['processor'].transform(test_data)
        

        test_data_win = get_sub_seqs(test_data, window_size=self.window_size, stride=1)
        test_loader = DataLoader(dataset=test_data_win, batch_size=self.batch_size, drop_last=False, pin_memory=True,
                                 shuffle=False)
        reconstr_errors, outputs_array = predict_test_scores(self.model, test_loader, latent=False,
                                                             return_output=True)
        if custom:
            predictions_dic = {'recons_err': reconstr_errors.mean(axis=1),
                                      'recons_vect': outputs_array,
                                      'anomalies' : None,
                                      'anomalies_score' : None
                                     }
            return predictions_dic
        
        else:
            anomalies, score = self.anomaly_vector_construction(reconstr_errors, norm=norm, sigma=sigma)

            if if_shap:
                return anomalies
            else:
                predictions_dic = {'recons_err': reconstr_errors,
                                   'recons_vect': outputs_array,
                                   'anomalies' : anomalies,
                                   'anomalies_score': score
                                   }
                return predictions_dic
            
def parse_arguments():
    """Command line parser.

    Returns:
        args: Arguments parsed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=10, 
                        help='The window size. Default is 10.')
    parser.add_argument('--contamination-ratio', type=float, default=0.0, 
                        help='The contamination ratio. Default is 0.')
    parser.add_argument('--seed', type=int, default=42, 
                        help='The random generator seed. Default is 42.')
    parser.add_argument('--save-dir', type=str, default='data/outputs',
                        help='The folder to store the model outputs.')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='The folder where the data are stored.')
    parser.add_argument('--if-mixed-data', action='store_false',
                        help='If datasets must be mixed or not. Default action is true.')
    return parser.parse_args()
            
def main(args):
    """Main function

    Args:
        args : Command-line arguments.
    """
    mixed_data = args.if_mixed_data
    contamination_ratio = args.contamination_ratio
    window_size = args.window_size
    seed = args.seed
    save_dir = args.save_dir
    data_dir = args.data_dir
    
    print('Data loading...')
    whole_train, test, normal_df, anomaly_df = get_data(data_dir)
    if mixed_data:
        data_random = data_processing_random(normal_df, anomaly_df, 
                                                         contamination_ratio=contamination_ratio, 
                                                         window_size=window_size, seed=seed)
    else:
        data_random = data_processing_naive(whole_train, test,
                                            contamination_ratio=contamination_ratio,
                                            window_size=window_size, seed=seed)
    
    model = AutoEncoder(name='auto_encoder', num_epochs=100, patience=20,
                        hidden_size=5, window_size=window_size, verbose=True,
                        save_dir=save_dir)
    start = time.time()
    model.fit(data_random['train']['data'], categorical_columns=None)
    end = time.time() - start
    print(f"\nTraining time = {end:.2f}\n")
    
    # Save the model
    save_torch_algo(model, save_dir=model.save_dir)
    print('Model saved !')
    
    # Prediction
    # Test
    print('\nBegin prediction...\n')
    y_true = data_random['test']['labels'][model.window_size]
    an_dict = model.predict(data_random['test']['data'], if_shap=False)
    an = an_dict['anomalies'][model.window_size]
    an_score = an_dict['anomalies_score'][model.window_size]
    _ = performance_evaluations(y_true=y_true, 
                                y_pred=an, 
                                y_score=an_score,
                                return_values=True,
                                plot_roc_curve=False)
    print("Prediction completed !")
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    