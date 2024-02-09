 #--command -- bash /project/src/bash/download.sh 

TAG=bruno_download

runai submit \
  --name  download-bruno \
  --image aicregistry:5000/${USER}:${TAG} \
  --backoff-limit 0 \
  --cpu-limit 4 \
  --gpu 0 \
  --large-shm \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/bruno/data/SHHS/:/data/ \
  --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project/  \ 
  --command -- sleep infinity \
  
runai submit --name download-bruno --image aicregistry:5000/wds20:bruno_download --backoff-limit 0 --cpu-limit 4 --gpu 0 --large-shm --host-ipc --project wds20 --volume /nfs/home/wds20/bruno/data/SHHS/:/data/ --volume /nfs/home/wds20/bruno/project/DDPM-EEG/:/project/  --command -- /usr/bin/expect /project/src/bash/download.exp
