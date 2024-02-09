TAG=ldm_eeg
config_file="/project/config/config_aekl_eeg.yaml"
path_pre_processed="/data/polysomnography/shhs_numpy"
type_dataset="shhs1"

channel="[32,32,64]"
specs=("no-spectral" "spectral")
latents=("1" "3")
# "2"
for sp in "${specs[@]}"; do
	for latent in "${latents[@]}"; do
	  modified_string=$(echo "$channel" | sed 's/\[//g; s/\]//g; s/,/-/g')
	  runai submit \
	    --name  metric-${modified_string}-${sp}-${latent} \
	    --image "aicregistry:5000/${USER}:${TAG}" \
	    --backoff-limit 0 \
	    --cpu-limit 150 \
	    --gpu 2 \
	    --node-type "A100" \
	    --large-shm \
	    --host-ipc \
	    --project wds20 \
	    --run-as-user \
      	    --volume /nfs/home/wds20/bruno/data/SHHS/shhs/:/data/ \
	    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
	    --command -- bash /project/src/bash/start_training.sh python3 /project/src/compute_mmds.py spe=${sp} latent=${latent} path_pre_processed=${path_pre_processed}
	done
done
