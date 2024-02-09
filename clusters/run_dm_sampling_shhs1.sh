TAG=ldm_eeg

specs=("no-spectral" "spectral")
dataset="shhsh"
path_pre_processed="/data/polysomnography/shhs_numpy"
for sp in "${specs[@]}"; do
  for indice in {0..1000..200}; do
    begin=$indice
    end=$((indice+200))
    echo "$begin $end"
    runai submit \
      --name dm-sample-shhs1-${sp}-${begin}-${end} \
      --image "aicregistry:5000/${USER}:${TAG}" \
      --backoff-limit 0 \
      --cpu-limit 25 \
      --gpu 1 \
      --node-type "A100" \
      --large-shm \
      --host-ipc \
      --project wds20 \
      --run-as-user \
      --volume /nfs/home/wds20/bruno/data/SHHS/shhs/:/data/ \
      --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
      --command -- bash /project/src/bash/start_training.sh python3 /project/src/testing/sample_trials_ddpm.py spe=${sp} dataset=${dataset} start_seed=${begin} stop_seed=${end}
	done
done
