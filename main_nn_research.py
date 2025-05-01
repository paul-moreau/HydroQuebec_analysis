import pandas as pd

from tqdm import tqdm

import torch

from dataset import load_dataset
from nn_fct import research_param

from itertools import product


x_train, y_train, x_test, y_test, x_val, y_val = load_dataset("data/final_datav2.csv",0.3,True)

# Grille d'hyperparamètres à tester
param_grid = {
    "width": [32, 64, 128],                # Largeur : 32, 64, 128 neurones par couche
    "depth": [2, 3, 4],                    # Profondeur : jusqu'à 4 couches cachées
    "learning_rate": [0.001, 0.01, 0.1],   # Taux d'apprentissage
    "batch_size": [16, 32, 64],            # Taille des batchs
}

# Variables pour enregistrer les meilleurs résultats
best_params = None
best_model = None

# Boucle principale : exploration des hyperparamètres
input_dim = x_train.shape[1]  # Nombre de colonnes/features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

total_runs = len(param_grid['width']) * len(param_grid['depth']) * len(param_grid['learning_rate']) * len(param_grid['batch_size'])
best_mse = float('inf')
best_mae = float('inf')
with tqdm(total=total_runs, desc='Grid Search Progress') as pbar:
    for width, depth, learning_rate, batch_size in product(param_grid['width'], param_grid['depth'], param_grid['learning_rate'], param_grid['batch_size']):
        tqdm.write(f"\nTest configuration: Width={width}, Depth={depth}, LR={learning_rate}, Batch size={batch_size}")

        mae, mse, model = research_param(x_train, y_train, x_test, y_test, input_dim, width, depth, learning_rate, batch_size)

        tqdm.write(f"MAE: {mae}")
        tqdm.write(f"MSE: {mse}")

        # Garder les meilleurs résultats
        if mse < best_mse:
            best_mse = mse
            best_mae = mae
            best_params = {'width': width, 'depth': depth, 'learning_rate': learning_rate, 'batch_size': batch_size}
            best_model = model
        
        pbar.update(1)

# Afficher les meilleurs hyperparamètres
print("Best Configuration:")
print(best_params)
print(f"Best MSE: {best_mse}")