ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER
# Alternatively, try tensorflow image
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dockerfiles/dockerfiles

LABEL maintainer="Ben Evans <ben.d.evans@gmail.com>"

# Install Tensorflow
RUN conda install --quiet --yes \
    'tensorflow-gpu=1.12*' \
    'keras=2.2*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
