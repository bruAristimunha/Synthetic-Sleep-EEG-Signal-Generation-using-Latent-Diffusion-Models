""" Script to compute the MS-SSIM score of the test set."""

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from braindecode.models import USleep
from monai.config import print_config
from monai.data import DataLoader, PersistentDataset
from monai.utils import set_determinism
from monai.transforms import Compose, LoadImageD, RandSpatialCropD, ScaleIntensityD, BorderPadD, EnsureChannelFirstD
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd
from pandas import DataFrame
from mne.filter import filter_data
from generative.metrics import FIDMetric
import mne
from collections import OrderedDict
from usleep import USleep
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from monai.transforms import Compose, LoadImageD, RandSpatialCropD, ScaleIntensityD, BorderPadD, EnsureChannelFirstD
from tqdm import tqdm
from omegaconf import OmegaConf
import pandas as pd
from pandas import DataFrame
from mne.filter import filter_data


mne.set_log_level('CRITICAL')

sfreq = 100
windows_size = 30 * sfreq
import warnings
#warnings.filterwarnings("ignore")

def get_transforms(dataset):
    if dataset == 'sleep_edfx':
        transforms_list = Compose([LoadImageD(keys='eeg'),
                                   ScaleIntensityD(factor=1e6, keys='eeg'),  # Numeric stability
                                   ScaleIntensityD(minv=0, maxv=1, keys='eeg'),  # Normalization
                                   RandSpatialCropD(keys='eeg', roi_size=[windows_size],
                                                    random_size=False, ),
                                   BorderPadD(keys='eeg', spatial_border=[36], mode="constant")
                                   ])

    elif dataset in ['shhs_h', 'shhs']:
        transforms_list = Compose([LoadImageD(keys='eeg'),
                                   EnsureChannelFirstD(keys='eeg'),
                                   ScaleIntensityD(factor=1e6, keys='eeg'),  # Numeric stability
                                   ScaleIntensityD(minv=0, maxv=1, keys='eeg'),  # Normalization
                                   RandSpatialCropD(keys='eeg', roi_size=[windows_size],
                                                    random_size=False, ),
                                   BorderPadD(keys='eeg', spatial_border=[36], mode="constant")
                                   ])
    return transforms_list

def get_datalist(
        df: DataFrame, basepath: str
):
    """
    Get data dicts for data loaders.

    """
    data_dicts = []
    for index, row in df.iterrows():

        # check if there is an .npy ending on the file name name, if not add it
        eeg_file_name = f"{row['FILE_NAME_EEG']}" if row['FILE_NAME_EEG'].endswith('npy') else f"{row['FILE_NAME_EEG']}.npy" 
        data_dicts.append(
            {
                "eeg": f"{basepath}/{eeg_file_name}",
                "subject": float(row["subject"]),
                "night": float(row["night"]),
                "age": float(row["age"]),
                "gender": str(row["gender"]),
                "lightoff": str(row["LightsOff"]),
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def test_dataloader(config, args, upper_limit=None):
    test_df = pd.read_csv(args.path_test_ids)

    if upper_limit is not None:
        test_df = test_df[:upper_limit]

    test_dicts = get_datalist(test_df, basepath=args.path_pre_processed)
    
    transforms_list = get_transforms(args.dataset)
    test_ds = PersistentDataset(data=test_dicts, transform=transforms_list,
                                 cache_dir=None)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=config.train.num_workers, drop_last=config.train.drop_last,
                              pin_memory=False,
                              persistent_workers=True)

    return test_loader
    

def get_datalist_syn(dataset, basepath):
    """
    Get data dicts for data loaders.

    """
    if dataset == "sleep_edfx":
        n = 64
    elif dataset == "shhs":
        n = 1000
        #n = 2288
    elif dataset == 'shhs_h':
        n = 132

    data_dicts = []
    for index in range(0, n):
        data_dicts.append(
            {
                "eeg": f"{basepath}/sample_{index}.npy",
                "participant_id": index,
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts   
    


def synthetic_dataloader(args, upper_limit=None):
    transforms_list = Compose([LoadImageD(keys='eeg')])
    test_dicts = get_datalist_syn(args.dataset, basepath=args.path_samples)

    test_ds = Dataset(data=test_dicts, transform=transforms_list)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                              num_workers=25, drop_last=False,
                              pin_memory=False,
                              persistent_workers=True)

    return test_loader

     
def computer_band(frequency, img_cropped, frequency_data):
    if frequency == "delta":
        img_cropped = filter_data(img_cropped.numpy().astype(float), sfreq=100, l_freq=0.5, h_freq=4.0)
    elif frequency == "theta":
        img_cropped = filter_data(img_cropped.numpy().astype(float), sfreq=100, l_freq=4.1, h_freq=8.0)
    elif frequency == "alpha":
        img_cropped = filter_data(img_cropped.numpy().astype(float), sfreq=100, l_freq=8.1, h_freq=12.0)
    elif frequency == "all":
        img_cropped = img_cropped.numpy()
    
    frequency_data[frequency] = img_cropped[0, 0, :].astype(float)
    
    return img_cropped


def computed_my_features(model, img_cropped, device):        
    img_cropped = img_cropped      
    img_double =  np.concatenate([img_cropped, img_cropped], 1)
    img_double = torch.from_numpy(img_double).type(torch.FloatTensor).to(device)
    model = model.to(device)
    with torch.no_grad():
        _, _, bottom = model(img_double)
        #outputs = torch.flatten(outputs, start_dim=1, end_dim=2)
    
    return bottom 
       
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", help="Path where to find the config file.")
    parser.add_argument("--path_test_ids", help="Location of file with test ids.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--path_pre_processed", type=str, help="Path to preprocessed data")
    parser.add_argument("--path_samples", help="Location to the generated samples.")
    parser.add_argument("--dataset", help="Dataset to be used",
            choices=["sleep_edfx", "shhs_h", "shhs"])
    parser.add_argument("--frequency", help="Frequency to be used",
        choices=["alpha", "theta", "delta", "all"])
    parser.add_argument("--i", help="part of the csv", type=int)
    parser.add_argument("--spe", help="part of the csv", type=str)
    args = parser.parse_args()
    return args


def main(args):
    print("Running analysis of test samples...")
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

    print("Getting data...")
    test_loader = test_dataloader(config=config, args=args, upper_limit=1000)
    test_loader_2 = synthetic_dataloader(args=args, upper_limit=1000)

    device = torch.device("cuda")
    n_channels=2
    n_classes=5
    sfreq=100

    model = USleep(
        in_chans=n_channels,
        sfreq=sfreq,
        depth=12,
        with_skip_connection=True,
        n_classes=n_classes,
        input_size_s=30,
        apply_softmax=False
    )
    state_dict = torch.load("/project/outputs/base_model/final_model.pth",
                        map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    frequencies = ['delta', 'theta', 'alpha', 'all']
    print("Computing FID...")
    frequency_data = {}
    frequency_data_b = {}
    acc_results = []
    frequency = args.frequency
    i = args.i
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    test_features = []
    samples_features = []

    test_features_b = []
    samples_features_b = []
    fid_list = []
    fid_list_b = []
    print(frequency)
    print("--------------------------------------------------")
    
    for step, batch in pbar:
        img = batch["eeg"]
        # remove the 0-padding
        img_cropped = img[:,:,36:-36]
                
        # Filter the data to include only one frequency
        img_cropped = computer_band(frequency, img_cropped, frequency_data)
        # Compute my feature space
        bottom = computed_my_features(model, img_cropped, device)
       
        samples_features.append(bottom.cpu())
    samples_features = torch.vstack(samples_features)     
    
    print("--------------------------------------------------")   
    pbar2 = tqdm(enumerate(test_loader_2), total=len(test_loader_2))
    for step, batch2 in pbar2:
        img2 = batch2["eeg"]
        img2_cropped = img2.squeeze(0)

        # Filter the data to include only one frequency
        img2_cropped = computer_band(frequency, img2_cropped, frequency_data_b)    
        # Compute my feature space
        bottom_test = computed_my_features(model, img2_cropped, device=device)    

        test_features.append(bottom_test.cpu())
        
    #import pdb; pdb.set_trace()
    test_features = torch.vstack(test_features)
      
    print("--------------------------------------------------")   

    with open(f'/project/outputs/fid_train_test_3/{args.spe}_{args.dataset}_{frequency}_1_{i}.pickle', 'wb') as handle:
        pickle.dump(samples_features, handle)
    with open(f'/project/outputs/fid_train_test_3/{args.spe}_{args.dataset}_{frequency}_2_{i}.pickle', 'wb') as handle:
        pickle.dump(test_features, handle)
    print("--------------------------------------------------")   
    print("over")
            




if __name__ == "__main__":
    args = parse_args()
    main(args)
