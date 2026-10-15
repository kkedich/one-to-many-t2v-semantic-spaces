ARG BASE_IMAGE
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.authors="karina.bogdan@gmail.com"

ARG UID
ARG GID
ARG USER_NAME
ARG GROUP

RUN echo $UID
RUN echo $GID
RUN echo $USER_NAME
RUN echo $GROUP

RUN echo ${USER_NAME}
RUN echo ${GROUP}

RUN echo ${USER_NAME}

RUN addgroup --gid $GID ${USER_NAME}
RUN adduser --disabled-password --gecos '' --uid $UID --gid $GID ${USER_NAME}
RUN adduser ${USER_NAME} sudo

RUN chmod 777 /home/${USER_NAME} \
    && ls -la /home/${USER_NAME}

USER $USER_NAME

ENV PATH="${PATH}:/home/${USERNAME}/.local/bin"
ENV PYTHONPATH="/home/${USERNAME}/.local/bin:/home/${USERNAME}/.local/lib/python3.6/site-packages/:/work/text2video/:${PYTHONPATH}"
CMD ["/bin/bash"]
