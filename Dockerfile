FROM nvcr.io/nvidia/pytorch:22.06-py3

ARG USER_ID
ARG GROUP_ID
ARG USER
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

ENV MNE_USE_NUMBA=false

COPY requirements.txt .
RUN python3 -m pip install pip --upgrade
RUN pip3 install -r requirements.txt  --no-cache-dir
USER $USER_ID

RUN pip install wandb
RUN python3 -m pip install -U --no-cache-dir wandb
