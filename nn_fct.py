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

def research_param(x_train, y_train, x_test, y_test, input_dim, width, depth, learning_rate, batch_size, device='cpu'):
    """
    Entraîne et évalue un modèle pour une configuration d'hyperparamètres donnée.
    """
    # Convertir les données en tenseurs
    train_loader = convert_dataset(x_train, y_train, batch_size, device)
    test_loader = convert_dataset(x_test, y_test, batch_size, device)

    # Définir le modèle
    model = FullyConnectedNet(input_dim=input_dim, output_dim=1, width=width, depth=depth).to(device)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.MSELoss()  # Régression -> Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Boucle d'entraînement
    num_epochs = 100  # Vous pouvez ajuster ce nombre
    for epoch in range(num_epochs):
        model.train()  # Mode entraînement
        for batch_X, batch_y in train_loader:
            # Passer les données au modèle
            outputs = model(batch_X).squeeze(1)  # Éliminer la dimension inutile
            loss = criterion(outputs, batch_y)

            # Optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Évaluation sur les données de test
    model.eval()  # Mode évaluation
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_pred_batch = model(x_batch).squeeze(1).cpu().numpy()
            y_true_batch = y_batch.cpu().numpy()
            all_preds.append(y_pred_batch)
            all_trues.append(y_true_batch)

    # Concaténation de tous les batchs
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
        
    # Calculer l'erreur (MAE et MSE)
    mae_val = mean_absolute_error(y_true, y_pred)
    mse_val = mean_squared_error(y_pred, y_true)
    return mae_val, mse_val, model

def train_test(model, learning_rate, train_loader, test_loader, patience, num_epochs, device='cpu'):
    criterion = nn.MSELoss()  # Régression -> Mean Squared Error
    mae_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_test_loss = float('inf')
    best_model_state = None

    all_loss_training = []
    all_mae_training = []

    all_loss_test = []
    all_mae_test = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(x_batch).squeeze(1)
            loss = criterion(y_pred, y_batch)
            mae = mae_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            running_mae += mae.item() * x_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        all_loss_training.append(train_loss)
        train_mae = running_mae / len(train_loader.dataset)
        all_mae_training.append(train_mae)

        model.eval()
        test_loss = 0.0
        test_mae = 0.0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)

                y_pred = model(x_test).squeeze(1)
                loss = criterion(y_pred, y_test)
                mae = mae_fn(y_pred, y_test)

                test_loss += loss.item() * x_test.size(0)
                test_mae += mae.item() * x_test.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        all_loss_test.append(test_loss)
        test_mae = test_mae / len(test_loader.dataset)
        all_mae_test.append(test_mae)

        print(f"Epoch {epoch+1}/{num_epochs} - Train MSE: {train_loss:.6f} - Train MAE: {train_mae:.6f}")
        print(f"                             - Test MSE: {test_loss:.6f} - Test MAE: {test_mae:.6f}")

        # === Early stopping ===
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()  # Sauvegarde du meilleur état
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping déclenché à l'epoch {epoch+1} !")
                break

    return (model, best_model_state), (all_loss_training, all_mae_training), (all_loss_test, all_mae_test)

def validate_model(model, val_loader, device='cpu'):
    criterion = nn.MSELoss()  # Régression -> Mean Squared Error
    mae_fn = nn.L1Loss()

    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)

            y_pred = model(x_val).squeeze(1)
            loss = criterion(y_pred, y_val)
            mae = mae_fn(y_pred, y_val)

            val_loss += loss.item() * x_val.size(0)
            val_mae += mae.item() * x_val.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    val_mae = val_mae / len(val_loader.dataset)

    return val_loss, val_mae

def validate_analyze(model, val_loader, device='cpu'):
    criterion = nn.MSELoss()  # Régression -> Mean Squared Error
    mae_fn = nn.L1Loss()

    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    all_preds = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)

            y_pred = model(x_val).squeeze(1)
            all_preds.append(y_pred)
            loss = criterion(y_pred, y_val)
            mae = mae_fn(y_pred, y_val)

            val_loss += loss.item() * x_val.size(0)
            val_mae += mae.item() * x_val.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    val_mae = val_mae / len(val_loader.dataset)

    return all_preds
