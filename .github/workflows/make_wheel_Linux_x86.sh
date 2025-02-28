set -e -x

df -h
docker info
# to get more disk space
rm -rf /usr/share/dotnet &

export TF_NAME='tensorflow'

# if tensorflow version >= 2.6.0 and <= 2.15.9
if [[ "$TF_VERSION" =~ ^2\.(16)\.[0-9]+$ ]] ; then
  export BUILD_IMAGE="tfra/nosla-cuda12.3-cudnn8.9-ubuntu20.04-manylinux2014-python$PY_VERSION"
  export TF_CUDA_VERSION="12.3"
  export TF_CUDNN_VERSION="8.9"
  export TF_USE_LEGACY_KERAS=1
elif [[ "$TF_VERSION" =~ ^2\.(15)\.[0-9]+$ ]] ; then
  export BUILD_IMAGE="tfra/nosla-cuda12.2-cudnn8.9-ubuntu20.04-manylinux2014-python$PY_VERSION"
  export TF_CUDA_VERSION="12.2"
  export TF_CUDNN_VERSION="8.9"
elif [[ "$TF_VERSION" =~ ^2\.(14)\.[0-9]+$ ]] ; then
  export BUILD_IMAGE="tfra/nosla-cuda11.8-cudnn8.7-ubuntu20.04-manylinux2014-python$PY_VERSION"
  export TF_CUDA_VERSION="11.8"
  export TF_CUDNN_VERSION="8.7"
elif [[ "$TF_VERSION" =~ ^2\.(12|13)\.[0-9]+$ ]] ; then
  export BUILD_IMAGE="tfra/nosla-cuda11.8-cudnn8.6-ubuntu20.04-manylinux2014-python$PY_VERSION"
  export TF_CUDA_VERSION="11.8"
  export TF_CUDNN_VERSION="8.6"
elif [[ "$TF_VERSION" =~ ^2\.([6-9]|10|11)\.[0-9]+$ ]] ; then
  export BUILD_IMAGE="tfra/nosla-cuda11.2-cudnn8-ubuntu20.04-manylinux2014-python$PY_VERSION"
  export TF_CUDA_VERSION="11.2"
  export TF_CUDNN_VERSION="8.1"
elif [ $TF_VERSION == "2.4.1" ] ; then
  export BUILD_IMAGE='tfra/nosla-cuda11.0-cudnn8-ubuntu18.04-manylinux2010-multipython'
  export TF_CUDA_VERSION="11.0"
  export TF_CUDNN_VERSION="8.0"
elif [ $TF_VERSION == "1.15.2" ] ; then
  export BUILD_IMAGE='tfra/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010-multipython'
  export TF_CUDA_VERSION="10.0"
  export TF_CUDNN_VERSION="7.6"
else
  echo "TF_VERSION is invalid: $TF_VERSION!"
  exit 1
fi

echo "BUILD_IMAGE is $BUILD_IMAGE"
echo "TF_CUDA_VERSION is $TF_CUDA_VERSION"
echo "TF_CUDNN_VERSION is $TF_CUDNN_VERSION"

if [ -z $HOROVOD_VERSION ] ; then
  export HOROVOD_VERSION='0.28.1'
fi

# For TensorFlow version 2.13 or later:
export PROTOBUF_VERSION='3.19.6'
if [[ "$TF_VERSION" =~ ^2\.1[3-9]\.[0-9]$ ]] ; then
  export PROTOBUF_VERSION='4.23.4'
fi

docker system prune -f

DOCKER_BUILDKIT=1 docker build --no-cache \
    -f tools/docker/build_wheel.Dockerfile \
    --output type=local,dest=wheelhouse \
    --build-arg PY_VERSION \
    --build-arg TF_VERSION \
    --build-arg TF_NAME \
    --build-arg TF_NEED_CUDA \
    --build-arg TF_CUDA_VERSION \
    --build-arg TF_CUDNN_VERSION \
    --build-arg HOROVOD_VERSION \
    --build-arg BUILD_IMAGE \
    --build-arg NIGHTLY_FLAG \
    --build-arg NIGHTLY_TIME \
    --build-arg PROTOBUF_VERSION \
    ./
