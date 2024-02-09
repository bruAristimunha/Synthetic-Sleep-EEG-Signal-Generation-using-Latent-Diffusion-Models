TAG=ldm_eeg

runai submit \
    --name debug-ldm-eeg \
    --image  "aicregistry:5000/${USER}:${TAG}" \
    --backoff-limit 0 \
    --gpu 0.5 \
    --cpu 1 \
    --large-shm \
    --host-ipc \
    --project wds20 \
    --run-as-user \
  --volume /nfs/home/wds20/bruno/data/SHHS/shhs/:/data/ \
  --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project/  \
    --interactive -- sleep infinity
