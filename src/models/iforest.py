# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import time, argparse


from ..utils.algorithm_utils import save_torch_algo, Algorithm
from ..utils.data_processing import get_sub_seqs, get_train_data_loaders, data_processing_naive, data_processing_random, get_data
from ..utils.evaluation_utils import performance_evaluations

class IForest(Algorithm):
    def __init__(self, name='IForest', seed: int=None, n_estimators=100, contamination='auto',
                save_dir=None, max_features=1.0, window_size=10):
        """Isolation Forest algorithm for anomaly detection.

        Args:
            name (str, optional)            : Algorithm's name. Defaults to 'IForest'.
            seed (int, optional)            : Random seed. Defaults to None.
            n_estimators (int, optional)    : The number of base estimators. Defaults to 100.
            contamination (str, optional)   : The amount of contamination in the dataset.
                                              Defaults to 'auto'.
            save_dir ([type], optional)     : Folder to save the outputs. Defaults to None.
            max_features (float, optional)  : The numbers of samples to draw
                                              from X to train each base estimator. Defaults to 1.0.
            window_size (int, optional)     : The size of the window moving. Defaults to 10.
        """
        Algorithm.__init__(self, __name__, name, seed=seed)
        self.model = None
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.window_size = window_size
        self.save_dir = save_dir
        self.additional_params = {}
        self.init_params = {
            'contamination' : contamination,
            'n_estimators' : n_estimators,
            'max_features' : max_features,
            'window_size' : window_size,
            'save_dir' : save_dir
        }
        
        
    def convert_predictions(self,pred):
        """Convert IF predictions from {-1;1} ({outlier;inlier}) to {0;1} ({inlier;outlier}).

        Args:
            pred : A prediction of the model. {-1;1} ({outlier;inlier})

        Returns:
                   The prediction converted. {0;1} ({inlier;outlier})
        """
        return (-pred+1.0)/2
        
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
        #print(type(train_data))
        train_data_win = get_sub_seqs(train_data, window_size=self.window_size)
        
            

        train_data_win = train_data_win.reshape(train_data_win.shape[0],-1)            
        #print(train_data.shape)
        self.model = IsolationForest(n_estimators=self.n_estimators, contamination=self.contamination,
                                    max_features=self.max_features)
        print(f"Fitting {self.name} model")
        self.model.fit(train_data_win)
        print("Fitting done")

    def predict(self, test_data : pd.DataFrame, if_shap=True):
        """Predict on the test dataframe

        Args:
            test_data (pd.DataFrame): Test dataframe.
            if_shap (bool, optional): If Shap values is computed during prediction. Defaults to True.

        Returns:
            np.array: Test predictions.
        """
        # Process the data
        test_data = self.additional_params['processor'].transform(test_data)
        
        # Create the window
        test_data_win = get_sub_seqs(test_data, window_size=self.window_size) 
            
    
        # Create a 0-vector of size window_size
        padding = np.zeros(test_data_win.shape[1]-1)
        # concatenate channels
        test_data_win = test_data_win.reshape(test_data_win.shape[0],-1)
        #print(test_data_win.shape)
        
        # binary classification
        anomalies = self.convert_predictions(self.model.predict(test_data_win).reshape(-1))
        anomalies = np.concatenate([padding, anomalies])
        
        # binary score
        
        #print(anomalies.shape)
        #score_t = np.concatenate([padding, score_t])
        if if_shap:
            return anomalies
        else:
            score = (-1.0)*self.model.decision_function(test_data_win) #https://github.com/lukasruff/Deep-SVDD/blob/master/src/isoForest.py
                                                                        # Or use score_sample() function
            predictions_dict = {'anomalies': anomalies,
                                'anomalies_score' : np.concatenate([padding, score])
                               }
            return predictions_dict
        
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
    
    model = IForest(save_dir=save_dir)
    start = time.time()
    model.fit(data_random['train']['data'], categorical_columns=['height', 'width', 'freeze'])
    end = time.time() - start
    print(f"\nTraining time = {end:.2f}\n")
    
    # Save the model
    save_torch_algo(model, save_dir=model.save_dir, torch_model=False)
    print('Model saved !')
    
    # Prediction
    # Test
    y_true = data_random['test']['labels'][model.window_size:]
    an_dict = model.predict(data_random['test']['data'], if_shap=False)
    an = an_dict['anomalies'][model.window_size:]
    an_score = an_dict['anomalies_score'][model.window_size:]
    _ = performance_evaluations(y_true=y_true, 
                                y_pred=an, 
                                y_score=an_score,
                                return_values=True,
                                plot_roc_curve=False)
    print("Prediction completed !")
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)