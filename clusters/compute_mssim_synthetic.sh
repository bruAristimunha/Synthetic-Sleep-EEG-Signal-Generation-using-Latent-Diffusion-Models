TAG=ldm_eeg
config_file="/project/config/config_aekl_eeg.yaml"
datasets=("sleep_edfx" "shhs_h" "shhs")
spes=("no-spectral" "spectral")
latent_channels=1

for dataset in "${datasets[@]}"; do
    for spe in "${spes[@]}"; do 
        echo ${dataset} ${spe}
        if [[ "${dataset}" = "sleep_edfx" ]] ;then 
            data_volume="/nfs/home/wds20/bruno/data/sleep_edfx/"
            dataset_form="sleep-edfx"
            path_samples="/project/sample_synthetic/samples_ldm_${latent_channels}_${spe}_sleep_edfx"
        elif [[ "${dataset}" = "shhs_h" ]] ;then
            data_volume="/nfs/home/wds20/bruno/data/SHHS"
            dataset_form="shhs-h"
            path_samples="/project/sample_synthetic/samples_ldm_${latent_channels}_${spe}_shhs_h"
        elif [[ "${dataset}" = "shhs" ]] ;then
            data_volume="/nfs/home/wds20/bruno/data/SHHS"
            dataset_form="shhs"
            path_samples="/project/sample_synthetic/samples_ldm_${latent_channels}_${spe}_shhs1"
        fi
        job_name="mssim-sample-${dataset_form}-${spe}"
        echo ${job_name}



    runai submit \
        --name  ${job_name} \
        --image "aicregistry:5000/${USER}:${TAG}" \
        --backoff-limit 0 \
        --cpu-limit 150 \
        --gpu 1 \
        --node-type "A100" \
        --large-shm \
        --host-ipc \
        --project wds20 \
        --run-as-user \
        --volume ${data_volume}:/data/ \
        --volume /nfs/home/wds20/jessica/DDPM-EEG/:/project  \
        --command -- bash /project/src/bash/start_training.sh \
        python3 /project/src/testing/MSSIM_sample.py \
             config_file=${config_file} \
             path_samples=${path_samples}
     done
 done 
