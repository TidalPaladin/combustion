# vim: set filetype=dockerfile :

# Base image with Apex
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel as base
MAINTAINER tidalpaladin

# install nvidia apex
ADD ./docker/common/install_apex.sh install_apex.sh
RUN bash ./install_apex.sh
RUN rm install_apex.sh

VOLUME ["/app/data", "/app/outputs", "/app/conf"]

# copy and install combustion
COPY src/combustion /app/src/combustion
COPY setup.py /app
WORKDIR /app
RUN pip install -e .


# Release version
FROM base as release
ARG PROJECT
COPY src/$PROJECT /app/src/$PROJECT

USER 1000
ENTRYPOINT ["bash"]

# Development version
FROM base as dev
RUN pip install -e .[dev]

ARG PROJECT
COPY src/$PROJECT /app/src/$PROJECT
COPY ./tests /app/tests

USER 1000
ENTRYPOINT ["bash"]
CMD ["python", "-m", "pytest", "-n", "auto", "--dist=loadfile", "-s", "-v", "/app/tests/"]

