TAG=ldm_eeg
path_pre_processed="/data/polysomnography/shhs_numpy_h"
specs=("no-spectral" "spectral")
latents=("1" "3")
type_dataset="shhs_h"

for sp in "${specs[@]}"; do
	for latent in "${latents[@]}"; do
	runai submit \
	  --name  sampling-shhsh-${sp}-${latent} \
	  --image "aicregistry:5000/${USER}:${TAG}" \
	  --backoff-limit 0 \
	  --cpu-limit 25 \
	  --gpu 1 \
	  --node-type "A100" \
	  --large-shm \
	  --host-ipc \
	  --project wds20 \
	  	    --run-as-user \
		--volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
		--volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
	  --command -- bash /project/src/bash/start_training.sh python3 /project/src/sample_trials.py spe=${sp} latent=${latent} type_dataset=${type_dataset} path_pre_processed=${path_pre_processed}
	done
done
