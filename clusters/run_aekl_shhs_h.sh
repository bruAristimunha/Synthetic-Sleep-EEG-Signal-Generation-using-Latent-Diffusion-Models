TAG=ldm_eeg
config_file="/project/config/config_aekl_eeg.yaml"
path_train_ids="/project/data/ids_shhs/ids_shhs_h_train.csv"
path_valid_ids="/project/data/ids_shhs/ids_shhs_h_valid.csv"
path_pre_processed="/data/polysomnography/shhs_numpy_h"
type_dataset="shhs_h"

channel="[32,32,64]"
specs=("no-spectral" "spectral")
latents=("1" "2" "3")
for sp in "${specs[@]}"; do
	for latent in "${latents[@]}"; do
	  modified_string=$(echo "$channel" | sed 's/\[//g; s/\]//g; s/,/-/g')
	  runai submit \
	    --name  shhsh-${modified_string}-${sp}-${latent} \
	    --image "aicregistry:5000/${USER}:${TAG}" \
	    --backoff-limit 0 \
	    --cpu-limit 25 \
	    --gpu-memory 20G \
	    --large-shm \
	    --host-ipc \
	    --project wds20 \
	    --run-as-user \
      --volume /nfs/home/wds20/bruno/data/SHHS/shhs/:/data/ \
	    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
	    --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_autoencoderkl.py config=${config_file} num_channels=${channel} spe=${sp} latent=${latent} path_train_ids=${path_train_ids} path_valid_ids=${path_valid_ids} type_dataset=${type_dataset} path_pre_processed=${path_pre_processed}
	done
done
