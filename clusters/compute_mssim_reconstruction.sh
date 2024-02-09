TAG=ldm_eeg
config_file="/project/config/config_aekl_eeg.yaml"
datasets=("sleep_edfx" "shhs_h" "shhs")
spes=("no-spectral" "spectral")
latent_channels=3

for dataset in "${datasets[@]}"; do
    for spe in "${spes[@]}"; do 
        echo ${dataset} ${spe}
        if [[ "${dataset}" = "sleep_edfx" ]] ;then 
            path_test_ids="/project/data/ids/ids_sleep_edfx_cassette_double_test.csv"
            path_pre_processed="/data/physionet-sleep-data-npy"
            best_model_path="/models/aekl_eeg_{}_{}_double_sleep_edfx"
            data_volume="/nfs/home/wds20/bruno/data/sleep_edfx/"
            dataset_form="sleep-edfx"
        elif [[ "${dataset}" = "shhs_h" ]] ;then
            path_test_ids="/project/data/ids_shhs/ids_shhs_h_test.csv"
            path_pre_processed="/data/shhs/polysomnography/shhs_numpy_h"
            best_model_path="/models/aekl_eeg_{}_shhs_h_{}"
            data_volume="/nfs/home/wds20/bruno/data/SHHS"
            dataset_form="shhs-h"
        elif [[ "${dataset}" = "shhs" ]] ;then
            path_test_ids="/project/data/ids_shhs/ids_shhs_test.csv"
            path_pre_processed="/data/shhs/polysomnography/shhs_numpy"
            best_model_path="/models/aekl_eeg_{}_shhs1_{}"
            data_volume="/nfs/home/wds20/bruno/data/SHHS"
            dataset_form="shhs"
        fi
        job_name="mssim-recons-${dataset_form}-${spe}"
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
            --volume /nfs/home/wds20/bruno/project/DDPM-EEG/outputs/:/models \
            --command -- bash /project/src/bash/start_training.sh \
            python3 /project/src/testing/MSSIM_reconstruction.py \
                 config_file=${config_file} \
                 path_test_ids=${path_test_ids} \
                 path_pre_processed=${path_pre_processed} \
                 best_model_path=${best_model_path} \
                 spe=${spe} \
                 latent_channels=${latent_channels} \
                 dataset=${dataset}
    done
done
