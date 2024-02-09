TAG=ldm_eeg
config_file="/project/config/config_dm.yaml"
specs=("no-spectral" "spectral")

for ((i=0; i<${#specs[@]}; i++)); do
  sp="${specs[$i]}"
  runai submit \
    --name  dm-eeg-${sp}-2000e \
    --image "aicregistry:5000/${USER}:${TAG}" \
    --backoff-limit 0 \
    --cpu-limit 25 \
    --gpu 2 \
    --large-shm \
    --host-ipc \
    --project wds20 \
    --run-as-user \
    --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
    --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_pure_ldm.py config=${config_file} spe=${sp} dataset="edfx"
done
