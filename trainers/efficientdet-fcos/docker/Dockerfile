# vim: set filetype=dockerfile :

###########################################################
# 	Base image with Apex
###########################################################
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime as base
MAINTAINER tidalpaladin

RUN apt-get update && apt-get install -y git gcc

# cv2 dependencies
RUN apt-get -y install libglib2.0-0 libsm6 libxrender-dev libxext6

# needed to build hydra?
#RUN apt-get -y install openjdk-8-jre

VOLUME ["/app/data", "/app/outputs", "/app/conf"]


###########################################################
# 	Python external dependencies
###########################################################
RUN python -m pip install -U hydra-core>=1.0.0

# install combustion from pip to hit most external dependencies
RUN pip install combustion==0.1.0rc2


###########################################################
# 	Submodules and Dependencies
###########################################################
# add combustion dependency info
ADD medcog_preprocessing/combustion/setup.py /app/medcog_preprocessing/combustion/
ADD medcog_preprocessing/combustion/version.txt /app/medcog_preprocessing/combustion/
RUN mkdir -p /app/medcog_preprocessing/combustion/src/combustion && touch /app/medcog_preprocessing/combustion/src/combustion/version.py

# add active contour dependency info
ADD medcog_preprocessing/mammogram_active_contour/setup.py /app/medcog_preprocessing/mammogram_active_contour/

# add preprocessing dependency info
ADD medcog_preprocessing/setup.py /app/medcog_preprocessing/setup.py

# add classifier dependency info
ADD setup.py requirements.txt /app/

# dependency only install
WORKDIR /app/
RUN pip install -r requirements.txt

# install combustion submodule
ADD medcog_preprocessing/combustion/ /app/medcog_preprocessing/combustion/
RUN pip install --no-deps /app/medcog_preprocessing/combustion

# install active contour submodule
ADD medcog_preprocessing/mammogram_active_contour/ /app/medcog_preprocessing/mammogram_active_contour/
RUN pip install --no-deps /app/medcog_preprocessing/mammogram_active_contour

# install preprocessing submodule
ADD medcog_preprocessing/ /app/medcog_preprocessing/
RUN pip install --no-deps /app/medcog_preprocessing


###########################################################
# 	Release Version
###########################################################
FROM base as release
ARG PROJECT

# install classifier
ADD medcog_classifier setup.py requirements.txt /app/medcog_classifier/
RUN pip install --no-deps /app

USER 1000
ENTRYPOINT ["bash"]


###########################################################
# 	Development Version
###########################################################

# Development version
FROM base as dev
RUN pip install -e .[dev]

ARG PROJECT
COPY src/$PROJECT /app/src/$PROJECT
COPY ./tests /app/tests

USER 1000
ENTRYPOINT ["bash"]
CMD ["python", "-m", "pytest", "-n", "auto", "--dist=loadfile", "-s", "-v", "/app/tests/"]
