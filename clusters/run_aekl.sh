TAG=ldm_eeg


config_file="/project/config/config_aekl_eeg_2_2_4.yaml"

runai submit \
  --name  aekl-eeg-2-2-4 \
  --image "aicregistry:5000/${USER}:${TAG}" \
  --backoff-limit 0 \
  --cpu-limit 4 \
  --gpu 1 \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --run-as-user \
  --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
  --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
  --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_autoencoderkl.py config=${config_file}
  
  
config_file="/project/config/config_aekl_eeg_4_4_16.yaml"

runai submit \
  --name  aekl-eeg-4-4-16 \
  --image "aicregistry:5000/${USER}:${TAG}" \
  --backoff-limit 0 \
  --cpu-limit 4 \
  --gpu 1 \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --run-as-user \
  --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
  --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
  --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_autoencoderkl.py config=${config_file}


config_file="/project/config/config_aekl_eeg_4_16_32.yaml"

runai submit \
  --name  aekl-eeg-4-16-32 \
  --image "aicregistry:5000/${USER}:${TAG}" \
  --backoff-limit 0 \
  --cpu-limit 4 \
  --gpu 1 \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --run-as-user \
  --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
  --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
  --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_autoencoderkl.py config=${config_file}
