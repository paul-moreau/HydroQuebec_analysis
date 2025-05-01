import numpy as np

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, output_dim, width, depth):
        """
        Args:
        - input_dim (int): Nombre de features en entrée.
        - output_dim (int): Dimension de la sortie (par ex. 1 pour régression).
        - width (int): Nombre de neurones par couche cachée.
        - depth (int): Nombre de couches cachées.
        """
        super(FullyConnectedNet, self).__init__()
        layers = []

        # Première couche (entrée -> première couche cachée)
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        # Couches cachées (largeur = `width`, profondeur = `depth-1`)
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Dernière couche (dernière couche cachée -> sortie)
        layers.append(nn.Linear(width, output_dim))

        # Assembler les couches dans un réseau séquentiel
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
# Load dataframe, split them, and convert bool to 0 and 1
def load_dataset(path, test_size, verbose=False):
    df = pd.read_csv(path)

    x = df.drop(['Total (kWh)'], axis=1)
    y = df['Total (kWh)']

    x_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=test_size, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    if verbose:
        print(f"Taille de l'ensemble d'entraînement: {len(x_train)}")
        print(f"Taille de l'ensemble de validation: {len(x_val)}")
        print(f"Taille de l'ensemble de test: {len(x_test)}")

    x_train = x_train.astype({col: 'int32' for col in x_train.select_dtypes('bool').columns})
    x_test = x_test.astype({col: 'int32' for col in x_test.select_dtypes('bool').columns})
    x_val = x_val.astype({col: 'int32' for col in x_val.select_dtypes('bool').columns})

    return x_train, y_train, x_test, y_test, x_val, y_val

def convert_dataset(x,y,batch_size,device='cpu'):
    # Convertir les données en tenseurs
    x_tensor = torch.tensor(x.to_numpy(), dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32).to(device)

    # Charger les données dans DataLoader pour gérer les mini-batchs
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

