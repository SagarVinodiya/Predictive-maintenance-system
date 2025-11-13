import pandas as pd

df = pd.read_csv("/mnt/data/Dataset_Predective_Maintanance.csv")

# Look at target column values
print(df['Operational Status'].value_counts(dropna=False))
