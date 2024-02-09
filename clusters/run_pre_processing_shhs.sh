TAG=ddpm_eeg_shhs

datasets=("shhs1")
# "shhs1h" parts={0..30}

for data in "${datasets[@]}"; do
	for part in {0..40}; do

	  echo  ddpm-eeg-shhs-${part}-${data}
    runai submit \
      --name ddpm-eeg-shhs-${part}-${data} \
      --image aicregistry:5000/${USER}:ddpm_eeg_shhs \
      --backoff-limit 0 \
      --cpu-limit 4 \
      --gpu 0 \
      --large-shm \
      --host-ipc \
      --project wds20 \
      --volume /nfs/home/wds20/bruno/data/SHHS/shhs/:/data/ \
      --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project/  \
      --command -- bash /project/src/bash/start_training.sh python3 /project/src/preprocessing/convert_shhs.py type=${data} part=${part}
  done
done


#bash /project/src/bash/start_training.sh python3 /project/src/preprocessing/convert_shhs.py type="shhs1h" part="0"
