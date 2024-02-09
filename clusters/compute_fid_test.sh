TAG=ldm_eeg
config_file="/project/config/config_aekl_eeg.yaml"
datasets=("sleep_edfx" "shhs_h" "shhs")
latent_channels=1
spes=("no-spectral" "spectral")
frequencies=("delta" "alpha" "theta" "all")
for dataset in "${datasets[@]}"; do
for frequency in "${frequencies[@]}"; do 
    echo ${dataset} ${frequency}
    
    if [[ "${dataset}" = "sleep_edfx" ]] ;then 
	path_test_ids="/project/data/ids/ids_sleep_edfx_cassette_double_test.csv"
	path_pre_processed="/data/physionet-sleep-data-npy"
	data_volume="/nfs/home/wds20/bruno/data/sleep_edfx/"
	dataset_form="sleep-edfx"
    elif [[ "${dataset}" = "shhs_h" ]] ;then
	path_test_ids="/project/data/ids_shhs/ids_shhs_h_test.csv"
	path_pre_processed="/data/shhs/polysomnography/shhs_numpy_h"
	data_volume="/nfs/home/wds20/bruno/data/SHHS"
	dataset_form="shhs-h"
    elif [[ "${dataset}" = "shhs" ]] ;then
	path_test_ids="/project/data/ids_shhs/ids_shhs_test.csv"
	path_pre_processed="/data/shhs/polysomnography/shhs_numpy"
	data_volume="/nfs/home/wds20/bruno/data/SHHS"
	dataset_form="shhs"
    fi
    job_name="fid-test-${dataset_form}-${frequency}"
    echo ${job_name}

    runai submit \
    --name ${job_name} \
    --image "aicregistry:5000/${USER}:${TAG}" \
    --backoff-limit 0 \
    --cpu-limit 10 \
    --node-type "A100" \
    --gpu 0.5 \
    --large-shm \
    --host-ipc \
    --project wds20 \
    --run-as-user \
    --volume ${data_volume}:/data/ \
    --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project  \
    --command -- bash /project/src/bash/start_training.sh \
    python3 /project/src/testing/FID_test.py \
	 config_file=${config_file} \
	 path_test_ids=${path_test_ids} \
	 path_pre_processed=${path_pre_processed} \
	 dataset=${dataset} \
	 frequency=${frequency}
done
done
