
# Dockerfile to be used
GPU_DOCKER_FILE=text2video.Dockerfile

# Docker image variables
SVC_GPU=text2video_myuser
VERSION=latest
REGISTRY_URL=kbogdan

# https://hub.docker.com/r/tensorflow/tensorflow/tags?page=1&name=2.15.0
TAG_VERSION_TF_GPU=2.15.0-gpu-jupyter
OPENCV_VERSION=4.9.0

# Volumes
# Since the makefile is generally inside a 'docker' folder, we get
# the parent of current dir
MY_INSIDE_VOLUME_GPU=/work

# Container related
NAME_CONTAINER=kbogdan-tf215

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
	@echo "Building Base/OpenCV image..."
	@docker build \
	    --build-arg TAG_VERSION=$(TAG_VERSION_TF_GPU) \
	    --build-arg OPENCV_VERSION=$(OPENCV_VERSION) \
	    -t "kbogdan/opencv" -f "opencv.Dockerfile" .
	@echo "Building Text2Video image..."
	@docker build \
	    --build-arg BASE_IMAGE=kbogdan/opencv:latest \
	    -t "kbogdan/text2video" -f "text2video.Dockerfile" .
	@echo "Building credentials image..."
	@docker build \
	    --build-arg BASE_IMAGE=kbogdan/text2video:latest \
	    --build-arg UID=`id -u` \
	    --build-arg GID=`id -g` \
	    --build-arg USER_NAME=${USER} \
	    --build-arg GROUP=`id -ng ${USER}` \
	    -t "kbogdan/text2video_myuser" -f "credentials.Dockerfile" .
	@echo "\n[build] Delete old image versions..."
	@docker images|sed "1 d"|grep "<none> *<none>"|awk '{print $$3}'|sort|uniq|xargs docker rmi -f
run-container-new:
    # To use the saved_models, you can copy from work to the container's path
	@echo "[run] This one is ok..."
	@echo "[run] Running container for a gpu-based image..."
	@echo $(MY_OUTSIDE_VOLUME_GPU)
	@docker run -it $(docker_gpu) \
		--userns=host --ipc=host \
		-p 6011:6006 -p 8891:8888 \
		-v ${HOME}:$(MY_INSIDE_VOLUME_GPU) \
		--workdir=$(MY_INSIDE_VOLUME_GPU) \
		--name $(NAME_CONTAINER) $(REGISTRY_URL)/$(SVC_GPU):$(VERSION) /bin/bash
upload-image:
	@echo "[upload] Uploading gpu docker image..."
	@docker push $(REGISTRY_URL)/$(SVC_GPU)
clean:
	@echo "[clean] Cleaning docker images..."
	@docker rmi -f $(REGISTRY_URL)/$(SVC_GPU):$(VERSION)
