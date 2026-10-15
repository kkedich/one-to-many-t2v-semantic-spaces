# Image related
TAG_VERSION_TF_GPU=1.13.2-gpu-py3
# incompatibility with 1.15.5-gpu-py3
OPENCV_VERSION=4.5.5
IMAGE_NAME=kbogdan/kinects_i3d_v2

# Volumes
# Since the makefile is generally inside a 'docker' folder, we get
# the parent of current dir
MY_INSIDE_VOLUME_GPU=/work

# Container related
NAME_CONTAINER=kbogdan-kinects-i3d-exec

# Check the docker version to correctly set the gpu command
docker_version=$(shell docker version --format '{{.Server.APIVersion}}')
ifeq ($(shell expr $(docker_version) \>= 1.40), 1)
    docker_gpu=--gpus all
else
    docker_gpu=--runtime=nvidia
endif


build-image:
	@echo "[build] Building gpu docker image..."
	@echo `id -ng ${USER}`
	@echo "Building base image..."
	@docker build --progress=plain \
	    --build-arg TAG_VERSION=$(TAG_VERSION_TF_GPU) \
	    --build-arg UID=`id -u` \
	    --build-arg GID=`id -g` \
	    --build-arg USER_NAME=${USER} \
	    --build-arg GROUP=`id -ng ${USER}` \
	    -t $(IMAGE_NAME) -f "gpu.Dockerfile" .
	@echo "\n[build] Delete old image versions..."
	@docker images|sed "1 d"|grep "<none> *<none>"|awk '{print $$3}'|sort|uniq|xargs docker rmi -f
# Without entrypoint definition
run-container:
	@echo "[run] Running container for a gpu-based image..."
	@docker run -it $(docker_gpu) \
		--userns=host \
		-v ${HOME}:$(MY_INSIDE_VOLUME_GPU) \
		--workdir=$(MY_INSIDE_VOLUME_GPU) \
		--name $(NAME_CONTAINER) $(IMAGE_NAME) /bin/bash
