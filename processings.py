import pandas as pd
import numpy as np
def date_processing(data,verbose=False):
    data['ANNEE_MOIS'] = pd.to_datetime(data['ANNEE_MOIS'])

    # Création de deux nouvelles colonnes pour l'année et le mois
    data['ANNEE'] = data['ANNEE_MOIS'].dt.year
    data['MOIS'] = data['ANNEE_MOIS'].dt.month

    data_result = data.drop('ANNEE_MOIS', axis=1)
    if verbose: 
        print(data_result[['MOIS', 'ANNEE']].head())

    # Calcul des coordonnées sin et cos
    data_result['MOIS_sin'] = np.sin(2 * np.pi * data_result['MOIS'] / 12)
    data_result['MOIS_cos'] = np.cos(2 * np.pi * data_result['MOIS'] / 12)

    data_result.drop('MOIS', axis=1, inplace=True)

    if verbose: 
        print(data_result[['MOIS_sin', 'MOIS_cos']].head())

    return data_result
def merge_df(data_hydro, data_pop, verbose=False):
    df_pop_long = data_pop.melt(id_vars=['Code', 'REGION_ADM_QC_TXT'], 
                         value_vars=[str(an) for an in range(2016,2024)], 
                         var_name='ANNEE', 
                         value_name='population')
    df_pop_long['ANNEE'] = df_pop_long['ANNEE'].astype(int)

    if verbose: print(df_pop_long)

    data_result = pd.merge(
        data_hydro, 
        df_pop_long[['REGION_ADM_QC_TXT', 'ANNEE', 'population']], 
        how='left',
        left_on=['REGION_ADM_QC_TXT', 'ANNEE'],
        right_on=['REGION_ADM_QC_TXT', 'ANNEE']
    )
    
    # Conversion du format de la colonne 'population'
    # Exemple : "147,030" --> 147030 (int)
    data_result['population'] = data_result['population'].replace({',': ''}, regex=True).astype(float).round().astype('Int64')

    if verbose : 
        print(data_result)

        nb_nan = data_result['population'].isna().sum()
        print(f"Nombre de NaN pour dans la colonne population : {nb_nan}")
        if nb_nan == 0: print('Merge réussi')
        else : print(data_result[data_result['population'].isna()])

    return data_result

def process_data_pop(data, verbose=False):
    data.rename(columns={'Région administrative': 'REGION_ADM_QC_TXT'}, inplace=True)

    colonnes_a_conserver = [
        col for col in data.columns
        if not str(col).isdigit() or (2016 <= int(col) <= 2023)
    ] # Sélectionne les colonnes entre 2016 et 2023

    data = data.loc[:, colonnes_a_conserver].copy() # Conserve seulement les données avec les colonnes à conserver

    colonnes_a_convertir = ['2016', '2017', '2018', '2019', '2020','2021','2022','2023']

    for colonne in colonnes_a_convertir:
        data[colonne] = data[colonne].str.replace(',', '').astype(int)

    if verbose: print(data.head()) 

    return data
