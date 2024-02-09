TAG=ldm_eeg

runai submit \
    --name debug-ldm-eeg-edfxy \
    --image  "aicregistry:5000/${USER}:${TAG}" \
    --backoff-limit 0 \
    --gpu 0.5 \
    --cpu 10 \
    --large-shm \
    --host-ipc \
    --project wds20 \
    --run-as-user \
    --volume /nfs/home/wds20/bruno/data/sleep_edfx/:/data/ \
    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
    --interactive -- sleep infinity
 
config_file="/project/config/config_aekl_eeg.yaml"
channel="[32,32,64]"
sp="no-spectral"
latent="2"
bash /project/src/bash/start_training.sh python3 /project/src/train_autoencoderkl.py config=${config_file} num_channels=${channel} spe=${sp} latent=${latent}
