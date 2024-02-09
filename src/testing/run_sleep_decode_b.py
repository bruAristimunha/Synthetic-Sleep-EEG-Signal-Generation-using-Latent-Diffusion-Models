import argparse
import os

from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import preprocess, Preprocessor, scale
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf

from util import setup_run_dir

if os.path.exists('/project'):
    base_path = '/project/'
    os.environ["WANDB__EXECUTABLE"] = "/opt/conda/bin/python3"
    base_path_data = '/data/'
else:
    base_path = '/home/bru/PycharmProjects/DDPM-EEG/'
    base_path_data = base_path

import pandas as pd
from braindecode.preprocessing import create_windows_from_events
from sklearn.preprocessing import minmax_scale
import numpy as np
from braindecode.samplers import SequenceSampler
from numbers import Integral
from sklearn.utils import compute_class_weight
import torch
from torch import nn
from skorch.helper import predefined_split
from skorch.callbacks import Checkpoint, EpochScoring, LoadInitState, \
    WandbLogger, LRScheduler

from braindecode import EEGClassifier
from sklearn.metrics import confusion_matrix, classification_report
from braindecode.visualization import plot_confusion_matrix
from skorch.helper import SliceDataset
import wandb
import mne
from braindecode.models import SleepStagerChambon2018, TimeDistributed
mne.set_log_level('ERROR')
min_max_instance = lambda x: minmax_scale(x, axis=1)
import matplotlib.pylab as plt

def get_center_label(x):
    if isinstance(x, Integral):
        return x
    return x[np.ceil(len(x) / 2).astype(int)] if len(x) > 1 else x




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=f"{base_path}/config/config_sleep_stage_b.yaml",
        help="Path to config file with all the training parameters needed",
    )
    parser.add_argument(
        "--path_train_ids",
        type=str,
        default=f"{base_path}/data/ids/ids_sleep_edfx_cassette_double_train.csv",
    )

    parser.add_argument(
        "--path_valid_ids",
        type=str,
        default=f"{base_path}/data/ids/ids_sleep_edfx_cassette_double_valid.csv",
    )

    parser.add_argument(
        "--path_test_ids",
        type=str,
        default=f"{base_path}/data/ids/ids_sleep_edfx_cassette_double_test.csv",
    )

    args = parser.parse_args()
    return args



def main(args):
    config = OmegaConf.load(args.config_file)

    set_determinism(seed=config.train.seed)
    print_config()

    run_dir, resume = setup_run_dir(config=config, args=args,
                                    base_path=base_path)

    train_subject = pd.read_csv(args.path_train_ids)['subject'].unique()
    test_subject = pd.read_csv(args.path_test_ids)['subject'].unique()
    valid_subject = pd.read_csv(args.path_valid_ids)['subject'].unique()

    dataset = SleepPhysionet(
        subject_ids=train_subject.tolist() + valid_subject.tolist() + test_subject.tolist(),
        crop_wake_mins=30)

    high_cut_hz = 18
    preprocessors = [
        Preprocessor('pick', picks=['Fpz-Cz']),
        Preprocessor('load_data'),
        Preprocessor(scale, factor=1e6, apply_on_array=True),
        Preprocessor('filter', l_freq=None, h_freq=high_cut_hz, verbose=False),
    ]


    preprocess(dataset, preprocessors, n_jobs=-1)

    mapping = {  # We merge stages 3 and 4 following AASM standards.
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4
    }

    window_size_s = 30
    sfreq = 100
    window_size_samples = window_size_s * sfreq

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        preload=True,
        mapping=mapping
    )

    from sklearn.preprocessing import scale as standard_scale

    preprocess(windows_dataset, [Preprocessor(standard_scale, channel_wise=True)])
    split_ids = dict(train=train_subject,
                     valid=valid_subject,
                     test=test_subject)
    splits = windows_dataset.split(split_ids)

    train_set, valid_set, test_set = splits["train"], splits["valid"], splits[
        "test"]

    # Print number of examples per class
    print('Training examples: ', len(train_set))
    print('Validation examples: ', len(valid_set))
    print('Test examples: ', len(test_set))



    y_train = np.array(list(SliceDataset(train_set, 1)))
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train), y=y_train)

    n_classes = 5
    # Extract number of channels and time steps from dataset
    n_channels, input_size_samples = train_set[0][0].shape

    model = SleepStagerChambon2018(
        n_chans=n_channels,
        sfreq=sfreq,
        n_outputs=n_classes,
        apply_batch_norm=True,
        dropout=0.9,
        pad_size_s=0.1,
        return_feats=False,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Send model to GPU
    if torch.cuda.is_available():
        model.cuda()

    lr = 1e-3
    batch_size = 32
    n_epochs = 1500
    weight_decay = 0.001

    check_point = Checkpoint(dirname=run_dir/ "checkpoint_b" )
    load_state = LoadInitState(check_point)
    train_bal_acc = EpochScoring(
        scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
        lower_is_better=False)
    valid_bal_acc = EpochScoring(
        scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
        lower_is_better=False)

    wandb.login(key="d4c4b9c56bda8a814e122301ad70b0d38f014728")
    wandb_run = wandb.init(project="sleep-stage", dir=str(base_path))

    callbacks = [
        ('train_bal_acc', train_bal_acc),
        ('valid_bal_acc', valid_bal_acc),
        ("cp", check_point),
        ("load_state", load_state),
        ("wandb", WandbLogger(wandb_run)),
        ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]

    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__weight=torch.Tensor(class_weights).to(device),
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=weight_decay,
        train_split=predefined_split(valid_set),
        iterator_train__shuffle=True,
        iterator_train__num_workers=10,
        iterator_valid__num_workers=10,
        # using valid_set for validation
        optimizer__lr=lr,
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
        classes=np.unique(y_train),
        max_epochs=n_epochs,
    )
    # Model training for a specified number of epochs. `y` is None as it is already
    # supplied in the dataset.
    clf.fit(train_set, y=None)
    clf.train_split = None  # Avoid pickling the validation set

    # with open(run_dir/'skorch_final_object.pickle', 'wb') as handle:
    #    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    clf.save_params(
        f_params=str(run_dir / "final_model_params.pkl"),
        f_history=str(run_dir / "final_model_history.json"),
        f_criterion=str(run_dir / "final_model_criterion.pkl"),
        f_optimizer=str(run_dir / "final_model_optimizer.pkl"),
    )

    print("Saving PyTorch model...")
    torch.save(model.state_dict(),
               str(run_dir / "final_model.pth"))

    torch.save(clf.module.state_dict(),
               str(run_dir / "final_module_0_state.pth"))

    torch.save(clf.module_.state_dict(),
               str(run_dir / "final_module_1_state.pth"))

    y_true = list(SliceDataset(valid_set, idx=1))

    y_pred = clf.predict(valid_set)

    confusion_mat = confusion_matrix(y_true, y_pred)

    fig = plot_confusion_matrix(confusion_mat=confusion_mat,
                                class_names=['Wake', 'N1', 'N2', 'N3', 'REM'])
    plt.savefig(str(run_dir / "confusion_matrix.png"))
    wandb.log({"final_fig": wandb.Image(fig)})
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    args = parse_args()
    main(args)
