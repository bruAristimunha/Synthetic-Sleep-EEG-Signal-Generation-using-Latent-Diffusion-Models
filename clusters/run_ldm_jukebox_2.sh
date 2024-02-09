TAG=ldm_eeg
config_file="/project/config/config_ldm.yaml"
autoencoderkl_config_file_path="/project/config/config_aekl_eeg.yaml"
channel="[32,32,64]"
best_model_path_list=("/project/outputs/aekl_eeg_no-spectral_1_double_sleep_edfx" "/project/outputs/aekl_eeg_no-spectral_3_double_sleep_edfx" "/project/outputs/aekl_eeg_spectral_1_double_sleep_edfx" "/project/outputs/aekl_eeg_spectral_3_double_sleep_edfx")
specs=("no-spectral" "spectral")
latent_channels=("1" "3")

for ((i=0; i<${#specs[@]}; i++)); do
  sp="${specs[$i]}"
  for ((j=0; j<${#latent_channels[@]}; j++)); do
	  latent_channel="${latent_channels[$j]}"
    	  ind=$((i*2 + j))
   	  best_model_path="${best_model_path_list[$ind]}"
	  modified_string=$(echo "$channels" | sed 's/\[//g; s/\]//g; s/,/-/g')
	  echo ${modified_string}-${latent_channel}
	  runai submit \
	    --name  ldm-eeg-${modified_string}-${sp}-l-${latent_channel} \
	    --image "aicregistry:5000/${USER}:${TAG}" \
	    --backoff-limit 0 \
	    --cpu-limit 25 \
	    --gpu-memory 20G \
	    --large-shm \
	    --host-ipc \
	    --project wds20 \
	    --run-as-user \
	    --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
	    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
	    --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_ldm.py num_channels=${channel} spe=${sp} autoencoderkl_config_file_path=${autoencoderkl_config_file_path} best_model_path=${best_model_path} latent_channels=${latent_channel}
	done
done
