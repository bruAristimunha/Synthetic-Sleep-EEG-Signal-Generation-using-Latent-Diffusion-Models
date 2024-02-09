import pickle
from generative.metrics import FIDMetric
import pandas as pd
import numpy as np

frequencies = ['alpha', 'delta', 'theta', 'all']
datasets = ["sleep_edfx", "shhs_h", "shhs"]
spes = ['spectral', 'no-spectral']
acc = []
for spe in spes:
    print(spe)
    metrics = FIDMetric()
    for dataset in datasets:
        print(dataset)
        for frequency in frequencies:
            print(frequency)
            feature_1 = pd.read_pickle(f'/project/outputs/fid_train_test/{spe}_{dataset}_{frequency}_1_0.pickle')
            feature_2 = pd.read_pickle(f'/project/outputs/fid_train_test/{spe}_{dataset}_{frequency}_2_0.pickle')
            print("load")
            #import pdb; pdb.set_trace()
            try:
                fid = metrics(feature_1.squeeze(-1), feature_2.squeeze(-1))
                print(f"{dataset} {frequency} {fid:.3f}")
                acc.append([dataset, frequency, spe, fid.numpy()])
            except:
                print(f"Error with {dataset} {frequency} ")
                
df = pd.DataFrame(acc, columns=['dataset', 'frequency', 'spe', 'fid'])
df['fid'] = df['fid'].astype(np.float64).round(4)
df.to_csv(f'/project/fid_train.csv', index=False)
print("AAAAAAAAAAAAAAAA")
