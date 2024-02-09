TAG=ldm_eeg
config_file="/project/config/config_ldm.yaml"

runai submit \
  --name  ldm-eeg \
  --image "aicregistry:5000/${USER}:${TAG}" \
  --backoff-limit 0 \
  --cpu-limit 25 \
  --gpu 2 \
  --node-type "A100" \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --run-as-user \
  --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
  --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
  --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_ldm.py config=${config_file}


#bash /project/src/bash/start_training.sh python3 /project/src/train_ldm.py config="/project/config/config_ldm.yaml"
