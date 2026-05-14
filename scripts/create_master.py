import pandas as pd
s = pd.read_csv('mimic-cxr-2.0.0-split.csv.gz', usecols=['subject_id', 'study_id', 'dicom_id', 'split'])
l = pd.read_csv('mimic-cxr-2.0.0-chexpert.csv.gz')
df = s.merge(l, on=['subject_id', 'study_id'])
df.to_csv('mimic_master_list.csv', index=False)
print("SUCCESS: Master list created.")
