TAG=ldm_eeg


runai submit \
  --name  filtering-eeg-sleep-edfx \
  --image "aicregistry:5000/${USER}:${TAG}" \
  --backoff-limit 0 \
  --cpu-limit 5 \
  --gpu 0 \
  --large-shm \
  --host-ipc \
  --project wds20 \
	--volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
	--volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
  --command -- python3 /project/src/preprocessing/convert_edfx.py \

