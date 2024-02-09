TAG=ldm_eeg
config_file="/project/config/config_ldm.yaml"
autoencoderkl_config_file_path="/project/config/config_aekl_eeg.yaml"
TAG=ldm_eeg
config_file="/project/config/config_aekl_eeg.yaml"
path_train_ids="/project/data/ids_shhs/ids_shhs_train.csv"
path_valid_ids="/project/data/ids_shhs/ids_shhs_valid.csv"
path_pre_processed="/data/polysomnography/shhs_numpy"
type_dataset="shhs1"

channel="[32,32,64]"
specs=("no-spectral" "spectral")
latents=("1" "3")

best_model_path_list=("/project/outputs/aekl_eeg_no-spectral_shhs1_1" "/project/outputs/aekl_eeg_no-spectral_shhs1_3" "/project/outputs/aekl_eeg_spectral_shhs1_1" "/project/outputs/aekl_eeg_spectral_shhs1_3")


for ((i=0; i<${#specs[@]}; i++)); do
  sp="${specs[$i]}"
  for ((j=0; j<${#latent_channels[@]}; j++)); do
	  latent_channel="${latent_channels[$j]}"
    	  ind=$((i*2 + j))
   	  best_model_path="${best_model_path_list[$ind]}"
	  modified_string=$(echo "$channels" | sed 's/\[//g; s/\]//g; s/,/-/g')
	  echo ${modified_string}-${latent_channel}
	  runai submit \
	    --name  ldm-shhs-${modified_string}-${sp}-l-${latent_channel} \
	    --image "aicregistry:5000/${USER}:${TAG}" \
	    --backoff-limit 0 \
	    --cpu-limit 25 \
	    --gpu-memory 1 \
	    --node-type "A100" \
	    --large-shm \
	    --host-ipc \
	    --project wds20 \
	    --run-as-user \
	    --volume /nfs/home/wds20/bruno/data/SHHS/shhs/:/data/ \
	    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
	    --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_ldm.py num_channels=${channel} spe=${sp} autoencoderkl_config_file_path=${autoencoderkl_config_file_path} best_model_path=${best_model_path} latent_channels=${latent_channel} path_train_ids=${path_train_ids} path_valid_ids=${path_valid_ids} type_dataset=${type_dataset}
	done
done
