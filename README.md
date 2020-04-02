# Combustion
This is a template repository for creating Dockerized deep learning projects.
It contains Docker files, a basic code skeleton, and some tests. The Tensorflow
code skeleton is **incomplete and untested** - it should be modified as needed for
your project. The `docker-compose.yml` and `.env` files should also be modified
as needed.

## Usage

Customizing Docker is done primarily through the `config.env` file. 

In this file you must specify:
- `DATA_DIR` - path to data, will be mounted at `/data` in the container
- `ARTIFACT_DIR` - path to runtime artifacts, will be mounted at `/artifacts` in the container
- `UPSTREAM` - upstream docker image that will be used as a base
- `APP_NAME` - name to assign to the built image

The container can be built with 

```sh
make build
```

Since there are a potentially large number of command line flags that 
must be specified for a training / testing run, these flags are passed
to the Makefile via `.env` files. A trivial `train.env` file is provided.
Multiple `.env` files can be created as needed and are passed to the 
Makefile like so

```
make runtime_conf="train.env" run
```

## Notes

The environment variables `DATA_DIR`, `ARTIFACT_DIR`, and `TEMP_DIR` will be
available inside the container. These environment variables are set to the
mount points of each respective volume within the container. The `TEMP_DIR` 
mountpoint is attached to a persistent volume than can be used to store items
like downloaded datasets or models that should persist between runs.

## To Do
- Add additional command line flags to Python code as needed
- More tests
- Add tests for building Docker container if possible
- Extract build artifacts from github action (codecov, images, etc.)
