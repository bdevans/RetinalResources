# NVIDIA Docker

## Build
docker build -t jktfg .

## Run
docker run --rm -p 8888:8888 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e JUPYTER_ENABLE_LAB=yes -v "$(pwd)":/home/jovyan/work jktfg
