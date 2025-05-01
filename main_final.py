import torch

import pandas as pd

from nn_fct import FullyConnectedNet, convert_dataset, load_dataset, validate_analyze
from processings import unnormalize, date_unprocessing

x_train, y_train, x_test, y_test, x_val, y_val = load_dataset("data/final_datav2.csv",0.3,True)

batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

val_loader = convert_dataset(x_val, y_val, batch_size, device)

model = FullyConnectedNet(
    input_dim=x_train.shape[1],
    output_dim=1,
    width=32,
    depth=4,
).to(device)

model.load_state_dict(torch.load("best_model.pth"))

all_preds = validate_analyze(
    model,
    val_loader=val_loader,
    device=device
)

y_pred = torch.cat(all_preds)
y_pred = y_pred.cpu().numpy()

y_pred_series = pd.Series(y_pred, index=x_val.index, name="Total (kWh) prédit")

df_val_with_preds = pd.concat([x_val, y_val, y_pred_series], axis=1)
df_val_with_preds = df_val_with_preds.rename(columns={y_val.name: "Total (kWh)"})

df_val_without_preds = pd.concat([x_val, y_val], axis=1)
df_val_without_preds = df_val_without_preds.rename(columns={y_val.name: "Total (kWh)"})

df_prenorm = pd.read_csv("data/full_data.csv")
print(df_val_with_preds.head())

columns_sector = [col for col in df_val_without_preds.columns if col.startswith('SECTEUR_')]
columns_region = [col for col in df_val_without_preds.columns if col.startswith('REGION_ADM_QC_TXT_')]

df_val_without_preds['SECTEUR'] = df_val_without_preds[columns_sector].idxmax(axis=1).str.replace('SECTEUR_', '')
df_val_without_preds['REGION_ADM_QC_TXT'] = df_val_without_preds[columns_region].idxmax(axis=1).str.replace('REGION_ADM_QC_TXT_', '')

df_val_with_preds['SECTEUR'] = df_val_with_preds[columns_sector].idxmax(axis=1).str.replace('SECTEUR_', '')
df_val_with_preds['REGION_ADM_QC_TXT'] = df_val_with_preds[columns_region].idxmax(axis=1).str.replace('REGION_ADM_QC_TXT_', '')

df_val_without_preds = df_val_without_preds.drop(columns=columns_sector + columns_region)
df_val_with_preds = df_val_with_preds.drop(columns=columns_sector + columns_region)

df_val_without_preds = unnormalize(df_prenorm.copy(deep=True), df_val_without_preds,preds=False, verbose=True)
df_val_with_preds = unnormalize(df_prenorm.copy(deep=True), df_val_with_preds,preds=True, verbose=True)

df_val_without_preds = date_unprocessing(df_val_without_preds,True)
df_val_with_preds = date_unprocessing(df_val_with_preds,True)


df_val_without_preds.to_csv("data/final_without_predictions.csv")
df_val_with_preds.to_csv("data/final_predictions.csv")


# print(len(all_preds))
# print(type(all_preds))
# print(all_preds[0].shape)
# print(type(all_preds[0]))
# print(all_preds[0])
# print(all_preds[-1])
# print(type(x_val))
# print(x_val.head())
# print(type(y_val))
# print(y_val.head())


