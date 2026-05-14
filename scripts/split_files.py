import pandas as pd
with open('mimic_train.csv', 'w') as train_file, open('mimic_val.csv', 'w') as val_file:
    first_chunk = True
    for chunk in pd.read_csv('mimic_master_list.csv', chunksize=50000):
        train_chunk = chunk[chunk['split'] == 'train']
        val_chunk = chunk[chunk['split'] == 'validate']
        if first_chunk:
            train_chunk.to_csv(train_file, index=False)
            val_chunk.to_csv(val_file, index=False)
            first_chunk = False
        else:
            train_chunk.to_csv(train_file, index=False, header=False)
            val_chunk.to_csv(val_file, index=False, header=False)
print("SUCCESS: mimic_train.csv and mimic_val.csv created.")
