import numpy as np
import pandas as pd
import time, argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from ..utils.algorithm_utils import save_torch_algo, Algorithm
from ..utils.data_processing import data_processing_naive, data_processing_random, get_data
from ..utils.evaluation_utils import performance_evaluations

class Anomaly_PCA(Algorithm):
    def __init__(self, name='PCA', seed: int=None, n_components=0.9, save_dir=None):
        """Anomaly PCA reconstruction algorithm for anomaly detection.

        Args:
            name (str, optional)            : Algorithm's name. Defaults to 'PCA'.
            seed (int, optional)            : Random seed. Defaults to None.
            n_components (float, optional)  : Number of principal components to keep.
                                              Defaults to 0.9.
            save_dir ([type], optional)     : Folder to save the outputs.
                                              Defaults to None.
        """
        Algorithm.__init__(self, __name__, name, seed=seed)
        self.model = None
        self.n_components = n_components
        self.init_params = {'n_components': n_components,
                            'save_dir' : save_dir
                            }
        self.additional_params = {}
        self.save_dir = save_dir

        

    def fit(self, train_data: pd.DataFrame, categorical_columns=None):
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
            processor = numerical_processor
            
        # Fit the model and get error reconstruction on the training datasets 
        self.model = Pipeline([('processor', processor),
                          ('pca', PCA(n_components=self.n_components))
                         ])
        print("Fitting PCA model")
        #train_data = train_data.values
        self.model.fit(X=train_data)
        print("Fitting done !")
        
        #recons_train = np.dot(self.model.transform(train_data), self.model['pca'].components_) + self.model['pca'].mean_
        #recons_train = self.model['scaler'].inverse_transform(recons_train)
        recons_train = self.model.inverse_transform(self.model.transform(train_data))
        
        
        # Get the reconstruction error on the training datasets
        recons_error = (train_data.values - recons_train)**2
        
        # Save min and max error for normalization of test errors.
        self.additional_params['train_error_mean'] = np.mean(recons_error, axis=0)
        self.additional_params['train_error_std'] = np.std(recons_error, axis=None)

    
    def anomaly_vector_construction(self, test_recons_errors, norm=2, sigma=3,
                                    return_anomalies_score=False):
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
            return anomalies, test_norm if return_anomalies_score else anomalies
        except ValueError:
            print("The model has not been trained ! No train error mean or std !")


    def predict(self, test_data: pd.DataFrame, norm=2, sigma=3, if_shap=True, custom=False):
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
        recons = self.model.inverse_transform(self.model.transform(test_data))

        
        # Compute the reconstruction error
        if type(test_data) == pd.DataFrame:
            recons_error = (test_data.values - recons) ** 2
        else:
            recons_error = (test_data - recons) ** 2

        if custom:
            predictions_dict = {'recons_err' :  recons_error.mean(axis=1),
                                'recons_vect' : recons,
                                'anomalies' : None,
                                'anomalies_score' : None
            }

            return predictions_dict

        else:
            
            # Compute anomalies
            if if_shap:
                anomalies = self.anomaly_vector_construction(recons_error, norm=norm, sigma=sigma)
                return anomalies
            else:
                anomalies, anomalies_score = self.anomaly_vector_construction(recons_error, norm=norm, sigma=sigma, return_anomalies_score=True)
                predictions_dict = {#'recons_err': recons_error,
                                   #'recons_vect': recons,
                                   'anomalies' : anomalies,
                                    'anomalies_score' : anomalies_score
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
    
    model = Anomaly_PCA(save_dir=save_dir)
    start = time.time()
    model.fit(data_random['train']['data'], categorical_columns=None)
    end = time.time() - start
    print(f"\nTraining time = {end:.2f}\n")
    
    # Save the model
    save_torch_algo(model, save_dir=model.save_dir, torch_model=False)
    print('Model saved !')
    
    # Prediction
    # Test
    print('\nBegin prediction...\n')
    y_true = data_random['test']['labels']
    an_dict = model.predict(data_random['test']['data'], if_shap=False)
    an = an_dict['anomalies']
    an_score = an_dict['anomalies_score']
    _ = performance_evaluations(y_true=y_true, 
                                y_pred=an, 
                                y_score=an_score,
                                return_values=True,
                                plot_roc_curve=False)
    print("Prediction completed !")
    
if __name__ == '__main__':
    args = parse_arguments()
    main(args)