import pandas as pd
import numpy as np

from processings import date_processing, merge_df, process_data_pop, normalize, to_one_hot

data_hydro = pd.read_csv("data/data_hydroquebec_brut.csv")
data_pop = pd.read_csv("data/data_pop_brut.csv")

data_hydro = date_processing(data_hydro)
data_pop = process_data_pop(data_pop)

data_result = merge_df(data_hydro,data_pop,verbose=True)

#data_result.to_csv('data/data_merged_brut.csv',index=False)

final_data = normalize(data_result)
final_data = to_one_hot(final_data)

print(final_data)
print(final_data.isna().sum())
print(final_data.isna()==True)

final_data.to_csv('data/final_data.csv', index=False)




