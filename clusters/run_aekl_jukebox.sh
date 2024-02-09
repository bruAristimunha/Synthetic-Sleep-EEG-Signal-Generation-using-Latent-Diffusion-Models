TAG=ldm_eeg
config_file="/project/config/config_aekl_eeg_retraining.yaml"
n_num_channels=("[32,32,64]")
#, "[2,2,4]" "[64,128,128]" "[128,128,128]" "[256,256,256]"  "[64,64,64]"
specs=("no-spectral" "spectral")

for sp in "${specs[@]}"; do
	for channel in "${n_num_channels[@]}"; do
	  modified_string=$(echo "$channel" | sed 's/\[//g; s/\]//g; s/,/-/g')
	  runai submit \
	    --name  aekl-re-${modified_string}-${sp} \
	    --image "aicregistry:5000/${USER}:${TAG}" \
	    --backoff-limit 0 \
	    --cpu-limit 25 \
	    --gpu-memory 8G \
	    --large-shm \
	    --host-ipc \
	    --project wds20 \
	    --run-as-user \
	    --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
	    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
	    --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_autoencoderkl.py config=${config_file} num_channels=${channel} spe=${sp} dataset="edfx"
	done
done
