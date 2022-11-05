#Script para la instalacion de object_detection_api en google colaboratory

!pip install -U --pre tensorflow=="2.*"
!pip install tf_slim
!pip install pycocotools

import os
import pathlib
import shutil

if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  !git clone --depth 1 https://github.com/tensorflow/models

%cd /content/models/research/
!protoc object_detection/protos/*.proto --python_out=.
shutil.copy("/content/models/research/object_detection/packages/tf2/setup.py","/content/models/research")
!pip install .