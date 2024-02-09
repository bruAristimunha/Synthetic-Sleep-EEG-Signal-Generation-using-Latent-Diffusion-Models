TAG=sleepdecode

runai submit \
  --name  sleep-decode-eeg-edf-b \
  --image "aicregistry:5000/${USER}:${TAG}" \
  --backoff-limit 0 \
  --cpu-limit 25 \
  --gpu 1 \
  --node-type "A100" \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --run-as-user \
  --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
  --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
  --command -- bash /project/src/bash/start_training.sh python3 /project/src/run_sleep_decode_b.py
