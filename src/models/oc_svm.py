import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import OneClassSVM
import time, argparse

from ..utils.algorithm_utils import save_torch_algo, Algorithm
from ..utils.data_processing import get_sub_seqs, get_train_data_loaders, data_processing_naive, data_processing_random, get_data
from ..utils.evaluation_utils import performance_evaluations


class OC_SVM(Algorithm):
    def __init__(self, name='OC_SVM', kernel='rbf', degree=3, gamma='auto',
        coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False,
        max_iter=-1, save_dir=None, window_size: int=10, seed: int=None):
        """One-Class SVM algorithm for anomaly detection.

        Args:
            name (str, optional)                    : Algorithm's name. Defaults to 'OC_SVM'.
            kernel (str, optional)                  : The kernel to use for the algorithm.
                                                      Defaults to 'rbf'.
            degree (int, optional)                  : The degree of polynomial kernel.
                                                      Defaults to 3.
            gamma (str, optional)                   : Kernel coefficient for kernel. 
                                                      Defaults to 'auto'.
            coef0 (float, optional)                 : Independent term in kernel. 
                                                      Defaults to 0.0.
            tol (float, optional)                   : Tolerance for stopping criterion. 
                                                      Defaults to 0.001.
            nu (float, optional)                    : The upper bound of anomalies in the dataset. 
                                                      Defaults to 0.5.
            shrinking (bool, optional)              : Whether the shrinking heuristic is used. 
                                                      Defaults to True.
            cache_size (int, optional)              : The size of the kernel cache. 
                                                      Defaults to 200.
            verbose (bool, optional)                : Defaults to False.
            max_iter (int, optional)                : The max number of iterations. 
                                                      Defaults to -1.
            save_dir ([type], optional)             : Folder to save the outputs. 
                                                      Defaults to None.
            window_size (int, optional)             : The size of the moving window. 
                                                      Defaults to 10.
            seed (int, optional)                    : Random seed. Defaults to None.
        """
        Algorithm.__init__(self, __name__, name, seed=seed)
        self.model = None
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.window_size = window_size
        #self.details = {}
        self.seed = seed
        self.save_dir = save_dir
        self.init_params = {"name": name,
                            "kernel": kernel,
                            "degree": degree,
                            "gamma": gamma,
                            "coef0": coef0,
                            "tol": tol,
                            "nu": nu,
                            "shrinking": shrinking,
                            "cache_size": cache_size,
                            "verbose": verbose,
                            "max_iter": max_iter,
                            "window_size": window_size,
                            "seed": seed,
                            "save_dir": save_dir,
                            }
        self.additional_params = dict()
        
        
    def convert_predictions(self,pred):
        """Convert OC-SVM predictions from {-1;1} ({outlier;inlier}) to {0;1} ({inlier;outlier}).

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
        #print(train_data.shape)
        train_data_win = get_sub_seqs(train_data, window_size=self.window_size)
        train_data_win = train_data_win.reshape(train_data_win.shape[0],-1)
        #print(train_data_win.shape)
        
        self.model = OneClassSVM(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                                 coef0=self.coef0, tol=self.tol, nu=self.nu, shrinking=self.shrinking,
                                 cache_size=self.cache_size, verbose=self.verbose, max_iter=self.max_iter)
        print("Fitting OC-SVM model")
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
        #print(test_data.shape)
        
        # Create the window
        test_data_win = get_sub_seqs(test_data, window_size=10)
        #print(test_data_win.shape)
            
    
        # Create a 0-vector of size window_size
        padding = np.zeros(test_data_win.shape[1]-1)
        # concatenate channels
        test_data_win = test_data_win.reshape(test_data_win.shape[0],-1)
        #print(test_data_win.shape)
        
        # binary score
        anomalies = self.convert_predictions(self.model.predict(test_data_win).reshape(-1))
        anomalies = np.concatenate([padding, anomalies])
        
        #print(anomalies.shape)
        #score_t = np.concatenate([padding, score_t])
        if if_shap:
            return anomalies
        else:
            score = self.model.decision_function(test_data_win)
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
    
    model = OC_SVM(save_dir=save_dir)
    start = time.time()
    model.fit(data_random['train']['data'], categorical_columns=['height', 'width', 'freeze'])
    end = time.time() - start
    print(f"\nTraining time = {end:.2f}\n")

    # Save the model
    save_torch_algo(model, save_dir=model.save_dir, torch_model=False)
    print('Model saved !')
    
    # Prediction
    # Test
    print('\nBegin prediction...\n')
    y_true = data_random['test']['labels'][model.window_size]
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