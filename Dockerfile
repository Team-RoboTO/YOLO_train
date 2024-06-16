FROM continuumio/miniconda3:latest

ARG LOCAL_PROJECT_PATH=/projects/Roboto
ARG CONTAINER_PROJECT_PATH=/projects/Roboto
ARG CONDA_ENV=yolov10

# COPY environment.yaml /tmp/environment.yaml

VOLUME /Roboto_volume

# SHELL ["conda", "create", "-n", "$CONDA_ENV"]
RUN conda create -n $CONDA_ENV python=3.9

# RUN conda activate ${CONDA_ENV}

CMD ["bash"]