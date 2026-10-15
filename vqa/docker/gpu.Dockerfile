# Dockerfile to run https://github.com/google-deepmind/kinetics-i3d
ARG TAG_VERSION
ARG BASE_IMAGE=tensorflow/tensorflow:${TAG_VERSION}
FROM ${BASE_IMAGE}
MAINTAINER kbogdan

# https://stackoverflow.com/a/25208687/10530728
RUN ln -s -f /bin/true /usr/bin/chfn

# Installing sonnet for the I3D model
# https://github.com/tensorflow/probability/releases/tag/v0.4.0
RUN pip install --no-deps tensorflow-probability==0.5.0 --no-cache-dir

# Copy the requirements file
COPY requirements.txt /tmp/

RUN pip install --upgrade pip setuptools --no-cache-dir && \
    pip install --requirement /tmp/requirements.txt --no-cache-dir


# Copy your entrypoint script (replace with actual path)
COPY entrypoint.sh /entrypoint.sh

# Set permissions for entrypoint script
RUN chmod +x /entrypoint.sh

# Cleaning temporary files and others
RUN apt-get autoclean autoremove &&\
    rm -rf /var/lib/apt/lists/* \
           /tmp/* \
           /var/tmp/*

ARG UID
ARG GID
ARG USER_NAME
ARG GROUP

RUN echo $UID
RUN echo $GID
RUN echo ${USER_NAME}
RUN echo ${GROUP}

RUN addgroup --gid $GID ${USER_NAME}
RUN adduser --disabled-password --gecos "" --uid ${UID} --gid ${GID} ${USER_NAME}
RUN adduser ${USER_NAME} sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN chmod 777 /home/${USER_NAME}
RUN ls -la /home/${USER_NAME}

USER $USER_NAME

ENV PATH="/home/${USERNAME}/.local/bin:$PATH"
ENV PYTHONPATH="/home/${USERNAME}/.local/bin:/home/${USERNAME}/.local/lib/python3.6/site-packages/:/work/text2video/:$PYTHONPATH"

CMD ["/bin/bash"]

# Define ENTRYPOINT script
ENTRYPOINT ["/entrypoint.sh"]
