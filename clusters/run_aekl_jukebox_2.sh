TAG=ldm_eeg
config_file="/project/config/config_aekl_eeg_retraining.yaml"
channel="[32,32,64]"
specs=("no-spectral" "spectral")
latents=("1" "3")
for sp in "${specs[@]}"; do
	for latent in "${latents[@]}"; do
	  modified_string=$(echo "channel" | sed 's/\[//g; s/\]//g; s/,/-/g')
	  runai submit \
	    --name  aekl-${modified_string}-${sp}-${latent} \
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
	    --command -- bash /project/src/bash/start_training.sh python3 /project/src/train_autoencoderkl.py config=${config_file} num_channels=${channel} spe=${sp} latent=${latent} dataset="edfx"
	done
done
