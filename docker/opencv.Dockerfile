ARG  TAG_VERSION
ARG BASE_IMAGE=tensorflow/tensorflow:${TAG_VERSION}
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.authors="karina.bogdan@gmail.com"


# Last stable version from
# https://opencv.org/releases/
ARG OPENCV_VERSION
ENV OPENCV_VERSION=${OPENCV_VERSION:-4.5.5}
ENV NUM_CORES=4
# https://askubuntu.com/a/1013396
ARG DEBIAN_FRONTEND=noninteractive


RUN pip install --no-cache-dir cmake


# Install OpenCV
RUN apt-get -y update -qq && \
    apt-get -y install wget \
                       unzip \
                       # Required
                       libatlas-base-dev \
                       libgtk2.0-dev \
                       libavcodec-dev \
                       libavformat-dev \
                       libswscale-dev \
                       # Optional
                       libtbb2 libtbb-dev \
                       libjpeg-dev \
                       libpng-dev \
                       libtiff-dev \
                       libv4l-dev \
                       #
                       # qt4-default \ Check if I need this
                       # Missing libraries for GTK
                       libatk-adaptor \
                       libcanberra-gtk-module \
                       # Tools
                       imagemagick \
                       # For use matplotlib.pyplot in python
                       python3-tk \
    && \
    apt-get autoclean autoremove &&\
    rm -rf /var/lib/apt/lists/* \
           /tmp/* \
           /var/tmp/*

WORKDIR /

    # Get OpenCV
RUN git clone https://github.com/opencv/opencv.git &&\
    cd opencv &&\
    git checkout $OPENCV_VERSION &&\
    cd / &&\
    # Get OpenCV contrib modules
    git clone https://github.com/opencv/opencv_contrib &&\
    cd opencv_contrib &&\
    git checkout $OPENCV_VERSION &&\
    mkdir /opencv/build &&\
    cd /opencv/build &&\
    #
    # Lets build OpenCV
    cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_NEW_PYTHON_SUPPORT=ON \
      -D BUILD_DOCS=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D WITH_TBB=ON \
      -D WITH_OPENMP=ON \
      -D WITH_IPP=ON \
      -D WITH_CSTRIPES=ON \
      -D WITH_OPENCL=ON \
      -D WITH_V4L=ON \
      .. &&\
    make -j$NUM_CORES \
    &&\
    make install \
    &&\
    ldconfig \
    &&\
    #
    # Clean the install from sources
    cd / &&\
    rm -r /opencv &&\
    rm -r /opencv_contrib &&\
    apt-get autoclean autoremove &&\
    rm -rf /var/lib/apt/lists/* \
           /tmp/* \
           /var/tmp/*
