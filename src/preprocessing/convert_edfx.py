from copy import deepcopy

import mne
import numpy as np
from pathlib import Path
from mne.datasets.sleep_physionet.age import fetch_data

folder_data = Path("/data/")
folder_data.mkdir(parents=True, exist_ok=True)

mne.set_config("PHYSIONET_SLEEP_PATH", folder_data.as_posix())

subject_ids = None
recording_ids = None
load_eeg_only = True
crop_wake_mins = 30

if subject_ids is None:
    subject_ids = range(83)
if recording_ids is None:
    recording_ids = [1, 2]

paths = fetch_data(
    subjects=subject_ids,
    recording=recording_ids, on_missing='warn')

ch_mapping = {
    'EOG horizontal': 'eog',
    'Resp oro-nasal': 'misc',
    'EMG submental': 'misc',
    'Temp rectal': 'misc',
    'Event marker': 'misc'
}
exclude = list(ch_mapping.keys()) if load_eeg_only else ()

for p in paths:
    raw = mne.io.read_raw_edf(p[0], preload=True, exclude=exclude)
    annots = mne.read_annotations(p[1])
    raw = raw.set_annotations(annots, emit_warning=False)
    # Rename EEG channels
    ch_names = {i: i.replace('EEG ', '') for i in raw.ch_names if 'EEG' in i}
    raw = raw.rename_channels(ch_names)
    mask = [
        x[-1] in ['1', '2', '3', '4', 'R'] for x in annots.description]
    sleep_event_inds = np.where(mask)[0]
    # Crop raw
    tmin = annots[int(sleep_event_inds[0])]['onset'] - crop_wake_mins * 60
    tmax = annots[int(sleep_event_inds[-1])]['onset'] + crop_wake_mins * 60
    raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))
    raw = raw.filter(l_freq=None, h_freq=18)

    path_annots = p[1].replace(".edf", "-annotation.npy").replace("physionet-sleep-data",
                                                                  "physionet-sleep-data-npy")
    print(f"Saving the annotation into {path_annots}")
    Path(path_annots).parent.mkdir(parents=True, exist_ok=True)
    annots = raw.annotations
    np.save(file=path_annots, arr=annots)

    for cha in raw.ch_names:
        path_npy = p[0].replace(".edf", f"-{cha}.npy").replace("physionet-sleep-data",
                                                               "physionet-sleep-data-npy")

        print(f"Saving the channel into {path_npy}")
        raw_pick = deepcopy(raw).pick_channels(ch_names=[cha])
        data = raw_pick.get_data()
        np.save(file=path_npy, arr=data)

print("Done")
