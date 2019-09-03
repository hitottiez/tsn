FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN    apt-get update \
    && apt-get -y install git libssl-dev wget curl lsb-release libopencv-dev build-essential checkinstall cmake libgtk2.0-dev \
    pkg-config yasm libdc1394-22 libdc1394-22-dev  \
    libjpeg-dev libpng12-dev libjasper-dev libavcodec-dev libavformat-dev \
    libswscale-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev \
    libv4l-dev libtbb-dev libqt4-dev libfaac-dev libmp3lame-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev \
    libxvidcore-dev x264 v4l-utils libxine2-dev libtiff5-dev libcurl4-gnutls-dev zlib1g-dev \
    python-dev python3-dev python3-pip python-pip \
    && apt-get upgrade -y curl libssl-dev \
    && apt-get clean

# cmake3.12インストール（boostのバージョンに合わせて）
WORKDIR /root/depends/cmake
RUN    wget https://cmake.org/files/v3.12/cmake-3.12.2.tar.gz \
    && tar -xf cmake-3.12.2.tar.gz \
    && cd cmake-3.12.2/ \
    && ./bootstrap --system-curl && make && make install

# PIPインストール
RUN    curl https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip3 install numpy scipy

WORKDIR /root/depends/opencv
RUN apt-get install zlib1g-dev
RUN apt-get install libcurl4-gnutls-dev

# glog, gflags, eigen3, 等々インストール
RUN    apt-get update \
    && apt-get -y install libgoogle-glog-dev sudo libopenblas-dev lsb-core libatlas-base-dev \
    && apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler \
    && apt-get -y install libgflags-dev libgoogle-glog-dev liblmdb-dev \
    && apt-get -y install libeigen3-dev \
    && pip3 install numpy protobuf \
    && ldconfig \
    && apt-get clean

# OpenCV 4.0.1 インストール
RUN    git clone --depth 1 -b 4.0.1 https://github.com/opencv/opencv.git \
    && git clone --depth 1 -b 4.0.1 https://github.com/opencv/opencv_contrib.git \
    && cd opencv \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
             -D CMAKE_INSTALL_PREFIX=/usr/local \
             -D BUILD_opencv_java=OFF \
             -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
             -D WITH_CUDA=ON \
             -D BUILD_TIFF=ON \
             -D BUILD_opencv_python3=ON \
             -D PYTHON3_EXECUTABLE=$(which python3) \
             -D WITH_TBB=ON \
             -D CPU_BASELINE=AVX2 \
             .. \
    && make -j$(nproc) \
    && make install

# boost1.68.0インストール（デフォルトの1.58ではboost::property_treeがthread safeではないので、たまにクラッシュする）
# python3のboost_pythonが必要なので、bootstrap.shにオプション追加
WORKDIR /root/depends/
RUN    wget https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.bz2 \
    && tar -xf boost_1_68_0.tar.bz2  \
    && cd boost_1_68_0/ \
    && sh bootstrap.sh --with-python=python3 --with-python-version=3.5 \
    && ./b2 install -j$(nproc) --prefix=/usr/local

ENV BOOST_ROOT /usr/local/

# Torchインストール
RUN pip3 install torch==0.3.1 torchvision==0.1.9 visdom tensorflow tensorboardX==1.1

WORKDIR /root/depends/flow_wrapper
COPY ./vendors/flow_wrapper /root/depends/flow_wrapper

RUN bash build.sh

RUN    apt-get update \
    && apt-get install -y language-pack-ja-base language-pack-ja
ENV LC_ALL ja_JP.UTF-8

RUN pip3 install pathlib pandas numba scikit-learn==0.21.1

RUN echo alias python='/usr/bin/python3.5' >> /root/.bashrc

WORKDIR /opt/multi_actrecog/tsn