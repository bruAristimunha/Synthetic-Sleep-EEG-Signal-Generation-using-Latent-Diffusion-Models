TAG=ldm_eeg
config_file="/project/config/config_dm.yaml"
path_train_ids="/project/data/ids_shhs/ids_shhs_h_train.csv"
path_valid_ids="/project/data/ids_shhs/ids_shhs_h_valid.csv"
path_pre_processed="/data/polysomnography/shhs_numpy_h"

specs=("no-spectral" "spectral")
for ((i=0; i<${#specs[@]}; i++)); do
  sp="${specs[$i]}"
  runai submit \
    --name  dm-shhsh-${sp}-2000e \
    --image "aicregistry:5000/${USER}:${TAG}" \
    --backoff-limit 0 \
    --cpu-limit 25 \
    --gpu 4 \
    --large-shm \
    --host-ipc \
    --project wds20 \
    --run-as-user \
      --volume /nfs/home/wds20/bruno/data/SHHS/shhs/:/data/ \
    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
    --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_pure_ldm.py config=${config_file} spe=${sp} path_train_ids=${path_train_ids} path_valid_ids=${path_valid_ids} path_pre_processed=${path_pre_processed} dataset="shhsh"
done
