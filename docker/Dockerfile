# vim: set filetype=dockerfile :

# Base image with Apex
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime as base
MAINTAINER tidalpaladin

RUN apt-get update && apt-get install -y git

VOLUME ["/app/data", "/app/outputs", "/app/conf"]

# copy and install combustion
COPY src/combustion /app/src/combustion
COPY setup.py /app
WORKDIR /app
RUN pip install -e .
COPY ./examples/basic /app/examples/basic

# Release version
FROM base as release
USER 1000
ENTRYPOINT ["bash"]

# Development version
FROM base as dev
RUN pip install -e .[dev]

COPY ./tests /app/tests

USER 1000
ENTRYPOINT ["bash"]
CMD ["python", "-m", "pytest", "-n", "auto", "--dist=loadfile", "-s", "-v", "/app/tests/"]

