import pickle
from generative.metrics import FIDMetric
import pandas as pd
import numpy as np

frequencies = ['alpha', 'delta', 'theta', 'all']
datasets = ["sleep_edfx", 'shhs_h']  # , 'shhs1']
for option in ['b', 'a']:
    print(option)
    metrics = FIDMetric()
    acc = []
    for dataset in datasets:
        print(dataset)
        for frequency in frequencies:
            print(frequency)
            if option == "a":
                option_type = ""
            else:
                option_type = "_b"
            feature_1 = pd.read_pickle(f'/project/outputs/fid_test/{dataset}_{frequency}_feature_1{option_type}.pickle')
            feature_2 = pd.read_pickle(f'/project/outputs/fid_test/{dataset}_{frequency}_feature_2{option_type}.pickle')
            print("load")
            #import pdb; pdb.set_trace()
            try:
                if option == "b":
                    fid = metrics(feature_1.squeeze(-1), feature_2.squeeze(-1))
                else:
                    fid = metrics(feature_1, feature_2)
                print(f"{dataset} {frequency} {fid:.3f}")
                acc.append([dataset, frequency, fid.numpy()])
            except:
                print(f"Error with {dataset} {frequency} ")
    df = pd.DataFrame(acc, columns=['dataset', 'frequency', 'fid'])
    import pdb; pdb.set_trace()
    df['fid'] = df['fid'].astype(np.float64).round(4)
    df.to_csv(f'/project/fid_test_{option}.csv', index=False)
    print("AAAAAAAAAAAAAAAA")
