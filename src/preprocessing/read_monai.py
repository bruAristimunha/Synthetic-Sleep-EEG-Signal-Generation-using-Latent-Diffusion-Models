import os
import tempfile
import numpy as np
import mne

from mne.datasets.sleep_physionet.age import fetch_data
from monai.data import Dataset
from monai.transforms import Compose, LoadImage, SpatialCropD, LoadImageD, IdentityD
from monai.utils.misc import first

from torcheeg.transforms import ToTensor, ChunkConcatenate


def getting_data(base_path, subject_ids=None, recording_ids=None):
    """
    Get data from mne.datasets.sleep_physionet.age.fetch_data
    :param subject_ids:
    :param recording_ids:
    :return: list of paths to npy files.
    """
    mne.set_config("PHYSIONET_SLEEP_PATH", base_path)

    if subject_ids is None:
        subject_ids = range(83)
    if recording_ids is None:
        recording_ids = [1, 2]

    paths = fetch_data(
        subjects=subject_ids,
        recording=recording_ids, on_missing='warn')
    eeg_signal = []
    for p in paths:
        file_save = p[0].replace(".edf", ".npy")
        eeg_signal.append({'eeg': file_save})

    return eeg_signal
eeg_signal = [{'eeg': ['/home/bru/PycharmProjects/DDPM-EEG/data/sleep-edfx/physionet-sleep-data/SC4011E0-PSG.npy','/home/bru/PycharmProjects/DDPM-EEG/data/sleep-edfx/physionet-sleep-data/SC4021E0-PSG.npy']}]
basepath = "/home/bru/PycharmProjects/DDPM-EEG/data/sleep-edfx"

subject_ids = [1, 2]
recording_ids = [1]
sfreq = 100

eeg_signal = getting_data(basepath, subject_ids, recording_ids)


eeg_transforms = Compose([LoadImageD(keys='eeg'),
                          SpatialCropD(keys='eeg', roi_slices=[slice(128), slice(128)])])

check_ds = Dataset(data=eeg_signal, transform=eeg_transforms)

eeg = first(check_ds)
print(eeg)

import ipdb; ipdb.set_trace()
# First attempt: LoadImage using the SpatialCrop transform.
"""

train_load = LoadImage()  # this is the one that works as instance
return_numpy_eeg = train_load(eeg_signal[0]['eeg'])
# We define a slice to crop the EEG signal, this slice have
crop_numpy = crop(return_numpy_eeg[0])

# It's not working as expected. It's cropping the signal, but not slicing the chunks.


crop = ChunkConcatenate([
    ToTensor()
],
    chunk_size=30 * sfreq,
    overlap=0),

read_crop = Compose([LoadImageD(keys='eeg'), crop])

trans = ChunkConcatenate(
    [IdentityD('eeg')],
    chunk_size=30 * sfreq,
    overlap=0)

train_load = LoadImageD('eeg')  # this is the one that works as dicionary

return_numpy = train_load(eeg_signal[0]['eeg'])

crop_2 = trans(return_numpy)
crop_numpy = crop(return_numpy[0])

# https://torcheeg.readthedocs.io/en/latest/torcheeg.transforms.html
print(return_numpy)

check_ds = Dataset(data=eeg_signal, transform=train_load)
eeg = first(check_ds)
print(eeg)

ransformation = Compose([LoadImage, trans])

eeg = np.load(eeg_signal[0]['eeg'])
"""