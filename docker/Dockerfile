# vim: set filetype=dockerfile :

# Base image with Apex
FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel as base
MAINTAINER tidalpaladin

# install nvidia apex
ADD ./docker/common/install_apex.sh install_apex.sh
RUN bash ./install_apex.sh
RUN rm install_apex.sh

VOLUME ["/app/data", "/app/outputs", "/app/conf"]
COPY src /app/src
COPY setup.py /app
WORKDIR /app
RUN pip install -e .

ENTRYPOINT ["bash"]

# Release version
FROM base as release
USER 1000

# Development version
FROM base as dev
COPY ./tests /app/tests
RUN pip install -e .[dev]

CMD ["python", "-m", "pytest", "-n", "auto", "--dist=loadfile", "-s", "-v", "/app/tests/"]

