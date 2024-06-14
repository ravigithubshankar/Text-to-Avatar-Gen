!pip install --upgrade --no-cache-dir gdown
!pip install streamlit
!pip install pyngrok
!pip install rembg[gpu] gradio==3.45.0

! wget -c "https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6.tar.gz"
! tar xf cmake-3.19.6.tar.gz
! cd cmake-3.19.6 && ./configure && make && sudo make install

# Install library
! sudo apt-get --assume-yes update
! sudo apt-get --assume-yes install build-essential
# OpenCV
! sudo apt-get --assume-yes install libopencv-dev
# General dependencies
! sudo apt-get --assume-yes install libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler
! sudo apt-get --assume-yes install --no-install-recommends libboost-all-dev
# Remaining dependencies, 14.04
! sudo apt-get --assume-yes install libgflags-dev libgoogle-glog-dev liblmdb-dev
# Python3 libs
! sudo apt-get --assume-yes install python3-setuptools python3-dev build-essential
! sudo apt-get --assume-yes install python3-pip
! sudo -H pip3 install --upgrade numpy protobuf opencv-python
# OpenCL Generic
! sudo apt-get --assume-yes install opencl-headers ocl-icd-opencl-dev
! sudo apt-get --assume-yes install libviennacl-dev

ver_openpose = "v1.7.0"
! echo $ver_openpose

! git clone  --depth 1 -b "$ver_openpose" https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# manually downloading openpose models
!%%bash
!gdown 1I4UWTXN5RjtmcmUy6oKUzdUAfD1haNYC
!unzip models.zip
!mv /content/models/face/pose_iter_116000.caffemodel /content/openpose/models/face/pose_iter_116000.caffemodel
!mv /content/models/hand/pose_iter_102000.caffemodel /content/openpose/models/hand/pose_iter_102000.caffemodel
!mv /content/models/pose/body_25/pose_iter_584000.caffemodel /content/openpose/models/pose/body_25/pose_iter_584000.caffemodel
!mv /content/models/pose/coco/pose_iter_440000.caffemodel /content/openpose/models/pose/coco/pose_iter_440000.caffemodel
!mv /content/models/pose/mpi/pose_iter_160000.caffemodel /content/openpose/models/pose/mpi/pose_iter_160000.caffemodel
!rm -rf models
!rm models.zip

! cd openpose && mkdir build && cd build

! cd openpose/build && cmake -DUSE_CUDNN=OFF -DBUILD_PYTHON=ON ..

! cd openpose/build && make -j`nproc`
! cd openpose && mkdir output
import os
%cd /content/
!rm -rf clothes-virtual-try-on
!git clone https://github.com/practice404/clothes-virtual-try-on.git
!mkdir /content/clothes-virtual-try-on/checkpoints

!gdown --id 18q4lS7cNt1_X8ewCgya1fq0dSk93jTL6 --output /content/clothes-virtual-try-on/checkpoints/alias_final.pth
!gdown --id 1uDRPY8gh9sHb3UDonq6ZrINqDOd7pmTz --output /content/clothes-virtual-try-on/checkpoints/gmm_final.pth
!gdown --id 1d7lZNLh51Qt5Mi1lXqyi6Asb2ncLrEdC --output /content/clothes-virtual-try-on/checkpoints/seg_final.pth

!gdown --id 1ysEoAJNxou7RNuT9iKOxRhjVRNY5RLjx --output /content/clothes-virtual-try-on/cloth_segm_u2net_latest.pth --no-cookies

%cd /content/
!pip install ninja

!git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing
%cd Self-Correction-Human-Parsing
!mkdir checkpoints

# downloading LIP dataset model
!gdown --id 1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH
!mv /content/Self-Correction-Human-Parsing/exp-schp-201908261155-lip.pth /content/Self-Correction-Human-Parsing/checkpoints/final.pth

import sys
_ = (sys.path
        .append("/usr/local/lib/python3.6/site-packages"))

!conda install --channel conda-forge featuretools --yes

!pip install opencv-python torchgeometry

!pip install torchvision

def make_dir():
  os.system("cd /content/ && mkdir inputs && cd inputs && mkdir test && cd test && mkdir cloth cloth-mask image image-parse openpose-img openpose-json")

from PIL import Image

import streamlit as st
from PIL import Image

def run(cloth, model):
  make_dir()
  cloth.save("/content/inputs/test/cloth/cloth.jpg")
  model.save("/content/inputs/test/image/model.jpg")

  # running script to compute the predictions
  os.system("rm -rf /content/output/")
  os.system("python /content/clothes-virtual-try-on/run.py")

  # loading output
  op = os.listdir("/content/output")[0]
  op = Image.open(f"/content/output/{op}")
  return op

# Set up the Streamlit app layout
st.title("Clothes Virtual Try ON")

col1, col2 = st.columns(2)

with col1:
    cloth_file = st.file_uploader("Upload the Cloth Image", type=["png", "jpg", "jpeg"])
    if cloth_file is not None:
        cloth_img = Image.open(cloth_file)
        st.image(cloth_img, caption="Cloth Image", use_column_width=True)
    else:
        cloth_img = None

with col2:
    model_file = st.file_uploader("Upload the Human Image", type=["png", "jpg", "jpeg"])
    if model_file is not None:
        model_img = Image.open(model_file)
        st.image(model_img, caption="Human Image", use_column_width=True)
    else:
        model_img = None

if st.button("Submit"):
    if cloth_img and model_img:
        final_output_img = run(cloth_img, model_img)
        st.image(final_output_img, caption="Final Prediction", use_column_width=True)
    else:
        st.warning("Please upload both images.")

# Hide Streamlit header and footer for a cleaner look
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
