import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        type=str,
        choices=["shhs1", "shhs1h"],
    )

    parser.add_argument(
        "--part",
        type=int,
        default=0
    )

    args = parser.parse_args()
    return args


def main(args):
    size_split = 1
    folder_data = Path("/data/polysomnography/")

    if args.type == "shhs1":
        save_dir = folder_data / "shhs_numpy/"
        file_name = "/project/data/ids_shhs/ids_sleep_shhs.csv"
    elif args.type == "shhs1h":
        save_dir = folder_data / "shhs_numpy_h/"
        file_name = "/project/data/ids_shhs/ids_sleep_shhs_h.csv"

    data_dir = (folder_data / "edfs/") / "shhs1/"
    ann_dir = (folder_data / "annotations-events-profusion/") / "shhs1/"

    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(file_name)

    # df = pd.read_csv("/project/data/ids_shhs/ids_sleep_shhs.csv")
    ids = df['nsrrid'].astype(str).values.tolist()

    edf_fnames = [os.path.join(data_dir, f"shhs1-{i}.edf") for i in ids]
    ann_fnames = [os.path.join(ann_dir, f"shhs1-{i}-profusion.xml") for i in ids]
    sav_fnames = [os.path.join(save_dir, f"shhs1-{i}-" + "{}.npy") for i in ids]

    edf_fnames = np.asarray(edf_fnames)
    ann_fnames = np.asarray(ann_fnames)
    sav_fnames = np.asarray(sav_fnames)

    edf_fnames_parts = np.array_split(edf_fnames, size_split)
    edf_fnames = edf_fnames_parts[args.part].tolist()

    ann_fnames_parts = np.array_split(ann_fnames, size_split)
    ann_fnames = ann_fnames_parts[args.part].tolist()

    sav_fnames_parts = np.array_split(sav_fnames, size_split)
    sav_fnames = sav_fnames_parts[args.part].tolist()
    list_wrong_files = list()
    for id_file, file_name in enumerate(tqdm(edf_fnames)):
        # Check if file is already processed
        if os.path.exists(sav_fnames[id_file].format("C4-A1")) and os.path.exists(sav_fnames[id_file].format("C3-A2")):
            print(f"processed {file_name}") 
        elif False:
            print(id_file, id_file/len(edf_fnames))
            raw = read_raw_edf(file_name, preload=True, stim_channel=None, verbose=False)
            raw = raw.filter(l_freq=None, h_freq=18)
            raw = raw.resample(100)
            sfreq = raw.info['sfreq']

            labels = []
            t = ET.parse(ann_fnames[id_file])
            r = t.getroot()
            faulty_File = 0
            for i in range(len(r[4])):
                lbl = int(r[4][i].text)
                if lbl == 4:  # make stages N3, N4 same as N3
                    labels.append(3)
                elif lbl == 5:  # Assign label 4 for REM stage
                    labels.append(4)
                else:
                    labels.append(lbl)
                if lbl > 5:  # some files may contain labels > 5 BUT not the selected ones.
                    faulty_File = 1

            if faulty_File != 1:
                print("============================== Faulty file ==================")
                continue

            labels = np.asarray(labels)
            y = labels.astype(np.int32)

            # Select on sleep periods
            w_edge_mins = 30
            nw_idx = np.where(y != 0)[0]
            start_idx = nw_idx[0] - (w_edge_mins * 2)
            end_idx = nw_idx[-1] + (w_edge_mins * 2)
            if start_idx < 0:
                start_idx = 0
            if end_idx >= len(y):
                end_idx = len(y) - 1

            select_chs = sorted(
                [s for s in raw.info["ch_names"] if "EEG" in s])
            # this has 2 vals [EEG,EEG(sec)] and selecting 0th index

            real_name_ch = ['C4-A1', 'C3-A2']

            for idx_ch, ch_name in enumerate(select_chs):
                # Cropped channelllll
                # print(real_name_ch[idx_ch])
                array = raw[ch_name][0].reshape(-1)[int(start_idx * sfreq * 30):
                                                    int((end_idx + 1) * sfreq * 30)]

                path_npy = sav_fnames[id_file].format(real_name_ch[idx_ch])
                print(path_npy)
                np.save(file=path_npy, arr=array)
        else:
            list_wrong_files.append(file_name)

    print("Done")
    print(list_wrong_files)


if __name__ == "__main__":
    args = parse_args()
    main(args)
