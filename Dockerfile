ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER
# Alternatively, try tensorflow image
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dockerfiles/dockerfiles

LABEL maintainer="Ben Evans <ben.d.evans@gmail.com>"

USER root
RUN apt-get install gnupg

# Add NVIDIA package repository
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
RUN apt install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
RUN apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
RUN apt update

# Install CUDA and tools. Include optional NCCL 2.x
RUN apt install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
    cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 \
    libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0

# Optional: Install the TensorRT runtime (must be after CUDA install)
RUN apt update
RUN apt install libnvinfer4=4.1.2-1+cuda9.0

# Install Tensorflow
RUN conda install --quiet --yes \
    'tensorflow-gpu=1.12*' \
    'keras=2.2*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID
