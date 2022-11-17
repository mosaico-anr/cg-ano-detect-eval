# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

import argparse, time

from src.models.anomaly_pca import Anomaly_PCA
from src.models.iforest import IForest
from src.models.oc_svm import OC_SVM
from src.models.auto_encoder import AutoEncoder
from src.models.lstm_vae import LSTM_VAE_Algo

from src.utils.algorithm_utils import save_torch_algo
from src.utils.data_processing import data_processing_naive, data_processing_random, get_data
from src.utils.evaluation_utils import performance_evaluations

available_models = ['pca', 'iforest', 'oc_svm', 'ae', 'lstm_vae']

def parse_arguments():
    """Command line parser.

    Returns:
        args: Arguments parsed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=available_models,
                        required=True, help='The model to train and test.')
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
    model_name = args.model_name
    mixed_data = args.if_mixed_data
    contamination_ratio = args.contamination_ratio
    window_size = args.window_size
    seed = args.seed
    save_dir = args.save_dir
    data_dir = args.data_dir

    if model_name not in available_models:
        raise ValueError(f"This model : {model_name} is not implemented.\n Must be in {available_models}.")
    
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
    
    if model_name == 'pca':
        model = Anomaly_PCA(save_dir=save_dir)
    elif model_name == 'iforest':
        model = IForest(save_dir=save_dir)
    elif model_name == 'oc_svm':
        model = OC_SVM(save_dir=save_dir)
    elif model_name == 'ae':
        model = AutoEncoder(name='auto_encoder', num_epochs=100, patience=20,
                    hidden_size=5, window_size=window_size, verbose=True,
                    save_dir=save_dir)
    elif model_name == 'lstm_vae':
        model = LSTM_VAE_Algo(name='lstm_vae', num_epochs=100, patience=20, lstm_dim=15,
                        hidden_size=3, save_dir=save_dir)
    else:
        raise ValueError(f"This model : {model_name} is not implemented.\n Must be in {available_models}.")

    # Training the model
    if model_name in ['iforest', 'oc_svm']:
        start = time.time()
        model.fit(data_random['train']['data'], categorical_columns=['height', 'width', 'freeze'])
        end = time.time() - start
    else:
        start = time.time()
        model.fit(data_random['train']['data'], categorical_columns=None)
        end = time.time() - start
    print(f"\nTraining time = {end:.2f}\n")


    # Save the model
    is_torch_model = (model_name in ['ae', 'lstm_vae'])
    save_torch_algo(model, save_dir=model.save_dir, torch_model=is_torch_model)
    print('Model saved !')
    
    # Prediction
    # Test
    print('\nBegin prediction...\n')
    if model_name == 'pca':
        y_true = data_random['test']['labels']
        an_dict = model.predict(data_random['test']['data'], if_shap=False)
        an = an_dict['anomalies']
        an_score = an_dict['anomalies_score']
    else:
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