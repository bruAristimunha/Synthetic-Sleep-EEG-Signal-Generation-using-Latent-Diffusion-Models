TAG=ldm_eeg
config_file="/project/config/config_ldm.yaml"
autoencoderkl_config_file_path="/project/config/config_aekl_eeg.yaml"
n_num_channels=("[32,32,64]")
best_model_path_list=("/project/outputs/aekl_eeg_32-32-64_no-spectral_10000_epochs-2048" "/project/outputs/aekl_eeg_32-32-64_spectral_10000_epochs-2048")
specs=("no-spectral" "spectral")

for ((i=0; i<${#specs[@]}; i++)); do
  sp="${specs[$i]}"
  for ((j=0; j<${#n_num_channels[@]}; j++)); do
	  channel="${n_num_channels[$j]}"
    ind=$((i*2 + j))
    best_model_path="${best_model_path_list[$ind]}"
	  modified_string=$(echo "$channel" | sed 's/\[//g; s/\]//g; s/,/-/g')
	  runai submit \
	    --name  ldm-eeg-${modified_string}-${sp}-2000h \
	    --image "aicregistry:5000/${USER}:${TAG}" \
	    --backoff-limit 0 \
	    --cpu-limit 25 \
	    --gpu-memory 10G \
	    --large-shm \
	    --host-ipc \
	    --project wds20 \
	    --run-as-user \
	    --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
	    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
	    --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_ldm.py config=${config_file} num_channels=${channel} spe=${sp} autoencoderkl_config_file_path=${autoencoderkl_config_file_path} best_model_path="/project/outputs/aekl_eeg_32-32-64_spectral_10000_epochs-2048"
	done
done
