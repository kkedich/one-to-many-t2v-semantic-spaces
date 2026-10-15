ARG BASE_IMAGE
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.authors="karina.bogdan@gmail.com"

ENV NUM_CORES=8

# Install requirements for the text2video project
RUN apt-get -y update -qq && \
    apt-get -y install \
                       # Required \
                       build-essential \
                       git \
                       pkg-config \
                       nano \
                       yasm \
                       vim \
                       # Java (for meteor)
                       default-jre \
                       ffmpeg && \
    # Cleaning temporary files and others
    apt-get autoclean autoremove &&\
    rm -rf /var/lib/apt/lists/* \
           /tmp/* \
           /var/tmp/*


# Install ffmpeg # =================================
# First install nvidia headers, now they use a separate repository for this
#WORKDIR /tmp/ffmpeg-codec
#
#RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git /tmp/ffmpeg-codec && \
#    cd /tmp/ffmpeg-codec && \
#    make && \
#    make install
#
#WORKDIR /
#WORKDIR /tmp/ffmpeg
#
## Removing --enable-libnpp, is generating an error. Original line: (--enable-cuvid  --enable-libnpp)
## probably will need an instalation of CUDA SDK
## https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.1/runtime/Dockerfile
## https://trac.ffmpeg.org/ticket/6405
#
## Compile and install ffmpeg from source
#RUN git clone https://github.com/FFmpeg/FFmpeg /tmp/ffmpeg && \
#  cd /tmp/ffmpeg && ./configure \
#  --enable-nonfree \
#  --disable-shared \
#  --enable-nvenc \
#  --enable-cuda \
#  --enable-cuvid  \
#  --extra-cflags=-I/usr/local/cuda/include \
#  --extra-cflags=-I/usr/local/include \
#  --extra-ldflags=-L/usr/local/cuda/lib64 && \
#  make -j$NUM_CORES && \
#  make install -j$NUM_CORES && \
#  cd /tmp && rm -rf ffmpeg
#
#WORKDIR /
## ===================================================

# Copy the requirements file
COPY requirements.txt /tmp/

RUN pip install --upgrade pip setuptools --no-cache-dir &&\
    pip install --requirement /tmp/requirements.txt --no-cache-dir &&\
    # Install packages for language model \
    pip install nltk --no-cache-dir &&\
    # https://github.com/heryandi/docker-python3-nltk-gensim/blob/master/Dockerfile \
    # python -m nltk.downloader -d /usr/local/share/nltk_data all \
    python3 -m nltk.downloader -d /usr/share/nltk_data punkt &&\
    python3 -m nltk.downloader -d /usr/share/nltk_data words
    # pip install --upgrade gensim

COPY DISTS/ /tmp/DISTS/

RUN cd /tmp/DISTS && \
    ls -lah && \
    # Clean existing build artifacts \
    rm -rf build/ dist/ *.egg-info/ && \
    pip install --use-pep517 --no-build-isolation .

CMD ["/bin/bash"]
