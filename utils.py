"""---------------------------------------------------------------------------------------------------
Este script contiene funciones prácticas/necesarias para la implementacion de modelos de deep learning 
en el campo de la vision artificial.
---------------------------------------------------------------------------------------------------"""

import os
import shutil
import random
import json


def rename_dataset(ruta):
    '''Renombra los archivos del dataset, cada archivo recibirá el nombre de la carpeta que 
    lo contine y un número
    
    Args:
        ruta: Recibe la ruta a la carpeta que contiene el dataset'''

    if os.path.isdir(ruta):
        for c,p in enumerate(os.listdir(ruta)):            
            global count,path
            count=c
            path=p
            ruta=os.path.join(ruta,path)
            rename_dataset(ruta)
            ruta=os.path.split(ruta)[0]              
    else:                
        carpeta=os.path.split(os.path.split(ruta)[0])[1]
        imagen=carpeta+str(count)+path[path.find("."):]
        #print("Convertimos: {} en: {}".format(ruta,os.path.join(os.path.split(ruta)[0],imagen)))
        os.rename(ruta,os.path.join(os.path.split(ruta)[0],imagen))
    
                
def split_dataset(ruta,train):
    """Renombra la carpeta original del dataset y se creará una carpeta que contendrá el 
    dataset train y test en función del % introducido en la función split_dataset
    
    Args:
        ruta: Recibe la ruta a la carpeta que contiene el dataset 
        train= float: Porcentaje de archivos que contendrá la carpeta train
    Return:
        orig_dataset/
        fnl_dataset/"""
    try:
        shutil.copytree(ruta,"orig_dataset")
    except:
        pass

    if os.path.isdir(ruta):
        ruta=ruta
        for p in os.listdir(ruta):
            global path
            path=p
            ruta=os.path.join(ruta,path)
            split_dataset(ruta,train)
            ruta=os.path.split(ruta)[0]
    else:
        os.makedirs("fnl_dataset/train",exist_ok=True)
        os.makedirs("fnl_dataset/test",exist_ok=True)     
        content=os.listdir(os.path.split(ruta)[0])
        for n in range(int(len(content)*train)):
            random_img=random.choice(content)
            shutil.move(os.path.join(os.path.split(ruta)[0],random_img),("fnl_dataset/train/"+random_img))
            content.remove(random_img)
        for m in content:
            shutil.move(os.path.join(os.path.split(ruta)[0],m),("fnl_dataset/test/"+m))
            content.remove(m)
            
    try:
        os.rmdir(ruta)
    except:
        pass


def json_to_csv(ruta):
    """Convierte el json MINI descargable del labelStudio y lo convierte en el formato TFRecord 
    para la implementación en modelos de tensorflow
    
    Args:
        ruta: Recibe la ruta al archivo json 
    Return:
        archivo.csv"""
    
    data=json.load(open(ruta))
    csv_list=[]

    for img in data:
        width, height = img["label"][0]["original_width"], img["label"][0]["original_height"]
        image=os.path.split(img["image"])[1]
        image=image[image.find("-")+1:]
        
        for i in img["label"]:
            name = i["rectanglelabels"][0]
            xmin = i["x"] * width / 100
            ymin = i["y"] * height / 100
            xmax = xmin + i["width"] * width / 100
            ymax = ymin + i["height"] * height / 100

            value = (image, width, height, name, xmin, ymin, xmax, ymax)
            csv_list.append(value)

    column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    csv_df = pd.DataFrame(csv_list, columns=column_name)
    csv_df.to_csv("labelStudio_{}.csv".format(ruta[ruta.find("_")+1:ruta.find(".")]))


def create_labelmap(csv_path):
    '''Funcion necesaria para la creación de los TFRecords, crea el label_map.pbtxt o en caso 
    de ya existir creará las clases del TFRecord basado en el labelmap
    
    Args:
        csv_path: Recibe la ruta al csv de train/test que será usado en la creación del label_map.pbtxt
    Return:
        label_map.pbtxt'''

    if not os.path.exists("/content/label_map.pbtxt"):
        label_dic = pd.DataFrame(pd.read_csv(csv_path))["class"].unique()
        label_dic = dict(enumerate(label_dic,start= 1))
        label_dic = dict(map(reversed, label_dic.items()))        
        with open("/content/label_map.pbtxt", "w") as f:
            for keys, values in label_dic.items():
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(keys))
                f.write('\tid:{}\n'.format(values))
                f.write('}\n')
    else:
        read_label_map()
    return label_dic

def read_label_map():
    item_id = None
    item_name = None
    label_dic = {}
    with open("/content/label_map.pbtxt", "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":")[1].replace("\"", " ")
                item_name = item_name.replace("'", " ").strip()   

            if item_id is not None and item_name is not None:
                label_dic[item_name] = item_id
                item_id = None
                item_name = None
    return label_dic


def inputs_colab():
    f = open ('inputs_modelo.txt','w')
    text='''
#-----------------------INSTALACION OBJECT_DETECTION_API-----------------------
#--------------(SOLO ES NECESARIO INSTALARLO 1 VEZ - En una celda aparte)--------------
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



#Para este scrip es necesario modificar las líneas indicadas y cargar en la raíz ("/content") los archivos **`"dataset.zip", "labelStudio_train.csv" y "labelStudio_test.csv"`**

#---------------------DEPENDENCIAS---------------------
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import sys
sys.path.append("../../models/research")

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

from google.colab import files

#---------------------FUNCIONES---------------------
def create_labelmap(csv_path):
    """Funcion necesaria para la creación de los TFRecords, crea el label_map.pbtxt o en caso 
    de ya existir creará las clases del TFRecord basado en el labelmap
    
    Args:
        csv_path: Recibe la ruta al csv de train/test que será usado en la creación del label_map.pbtxt
    Return:
        label_map.pbtxt"""

    if not os.path.exists("/content/label_map.pbtxt"):
        label_dic = pd.DataFrame(pd.read_csv(csv_path))["class"].unique()
        label_dic = dict(enumerate(label_dic,start= 1))
        label_dic = dict(map(reversed, label_dic.items()))        
        with open("/content/label_map.pbtxt", "w") as f:
            for keys, values in label_dic.items():
                f.write('item { \\n')
                f.write('\\tname:\\'{}\\'\\n'.format(keys))
                f.write('\\tid:{}\\n'.format(values))
                f.write('}\\n')
    else:
        label_dic= read_label_map()
    return label_dic

def read_label_map():
    item_id = None
    item_name = None
    label_dic = {}
    with open("/content/label_map.pbtxt", "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":")[1].replace("\\"", " ")
                item_name = item_name.replace("'", " ").strip()   

            if item_id is not None and item_name is not None:
                label_dic[item_name] = item_id
                item_id = None
                item_name = None
    return label_dic

def class_text_to_int(row_label, labelmap):
    if labelmap.get(row_label) != None:  
        return labelmap[row_label]
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tfrecords(train_csv_path, test_csv_path, compress_and_download=False):
    """Creación de los TFRecords de train y test a partir de las rutas de los .csv de train y
    test, con la opción de comprimirlos junto al label_map y descargarlos
    
    Args:
        train_csv_path: Recibe la ruta al archivo *train.csv producto del metodo dyno.json_to_csv
        test_csv_path: Recibe la ruta al archivo *test.csv producto del metodo dyno.json_to_csv
        compress_and_download= bool: Comprime y descarga un .zip con los archivos label_map.pbtxt train.record test.record
    Return:
        train.record
        test.record"""
                
    if not os.path.exists("/content/dataset"): 
        !unzip /content/dataset.zip -d /content/dataset

    if os.path.exists("/content/label_map.pbtxt"):
        output_path = "/content/train.record"
        path = "/content/dataset/dataset"    
        
        global label_map
        label_map = create_labelmap(train_csv_path)
        writer = tf.io.TFRecordWriter(output_path)
        examples = pd.read_csv(train_csv_path)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))


        output_path = "/content/test.record"
        path = "/content/dataset/dataset"     
        
        label_map = create_labelmap(test_csv_path)
        writer = tf.io.TFRecordWriter(output_path)
        examples = pd.read_csv(test_csv_path)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))
    else:
        create_labelmap(train_csv_path)
        create_tfrecords(train_csv_path, test_csv_path)
    
    if compress_and_download==True:
        %cd /content
        !zip inputs_model.zip label_map.pbtxt train.record test.record      #comprime los archivos creados por la función
        files.download("/content/inputs_model.zip")

#------------------------------------------------------------------------------------------
#----------------------------------- LÍNEAS A MODIFICAR -----------------------------------
#------------------------------------------------------------------------------------------
create_labelmap("/content/labelStudio_train.csv")
create_tfrecords("/content/labelStudio_train.csv","/content/labelStudio_test.csv", True)
#------------------------------------------------------------------------------------------
    '''
    f.write(text)
    f.close()


def train_colab():
    f = open ('entrenamiento_modelo.txt','w')
    text='''
#-----------------------INSTALACION OBJECT_DETECTION_API-----------------------
#--------------(SOLO ES NECESARIO INSTALARLO 1 VEZ - En una celda aparte)---------------
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


#Para este scrip es necesario modificar las líneas indicadas y cargar en la raíz ("/content") los archivos **`"label_map.pbtxt", "train.record" y "test.record" o por el unico archivo "inputs_model.zip"`** 
#---------------------DEPENDENCIAS---------------------
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from google.colab import files


#---------------------FUNCIONES---------------------
def download_model():
    !wget --no-check-certificate http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz \
        -O /content/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
    !tar -zxvf /content/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
    
    output_path = os.path.join(os.getcwd(),'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')

    os.remove("/content/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz")

    if not os.path.exists("/content/model"): os.mkdir("/content/model") 
    if not os.path.exists("/content/model/pipeline.config"): shutil.copyfile("{}/pipeline.config".format(output_path), "/content/model/pipeline.config")

def unzip_inputs():
    if os.path.exists("/content/inputs_model.zip"): 
        !unzip -q /content/inputs_model.zip -d /content

def read_label_map():
    item_id = None
    item_name = None
    label_dic = {}
    with open("/content/label_map.pbtxt", "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":")[1].replace("\\"", " ")
                item_name = item_name.replace("'", " ").strip()   

            if item_id is not None and item_name is not None:
                label_dic[item_name] = item_id
                item_id = None
                item_name = None
    return label_dic

def conf_pipeline(batch=8, num_steps=5000):
    """Función que configura el pipeline de entrenamiento.
    
    Parameters:
        batch= int: Número de imagenes que consumirá el modelo por ciclo
        num_steps= int: Número de ciclos ejecutados en el entrenamiento
    Return: 
        pipeline.config modificado

    (!!!En caso de que NO se haya descargado el modelo en el instante que se ejecuta este scrip, se ejecutará su descarga!!!)"""

    global num_stps
    num_stps=num_steps 

    if not os.path.exists("/content/label_map.pbtxt") or not os.path.exists("/content/train.record") or not os.path.exists("/content/test.record"): unzip_inputs()

    classes = len(read_label_map().keys())

    if not os.path.exists("/content/models/research/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint"): download_model()

    config = config_util.get_configs_from_pipeline_file("/content/model/pipeline.config")
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile("/content/model/pipeline.config", "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    pipeline_config.model.ssd.num_classes = classes
    pipeline_config.train_config.batch_size = batch
    pipeline_config.train_config.fine_tune_checkpoint = "/content/models/research/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0"
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = "/content/label_map.pbtxt"
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[0] = "/content/train.record"
    pipeline_config.eval_input_reader[0].label_map_path = "/content/label_map.pbtxt"
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[0] = "/content/test.record"

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile("/content/model/pipeline.config", "wb") as f:
        f.write(config_text)
        f.close()
    return

def train_model():
    """Función utilizada para entrenar el modelo siguiendo la configuración del pipeline.config, haciendo uso de los archivos 
    train.record, test.record y label_map.pbtxt.
    
    Parameters: 
        Any
    Return: 
        pipeline.config modificado"""
        
    !python /content/models/research/object_detection/model_main_tf2.py \\
    --pipeline_config_path={"/content/model/pipeline.config"} \\
    --model_dir={"/content/model"} \\
    --num_train_steps={num_stps}

def exporter_model(download=False):
    """Función utilizada para exportar el modelo.
    
    Parameters: 
        Any
    Return: 
        fnl_model.zip"""

    !python /content/models/research/object_detection/exporter_main_v2.py \\
    --input_type image_tensor \\
    --pipeline_config_path {"/content/model/pipeline.config"} \\
    --trained_checkpoint_dir {"/content/model"} \\
    --output_directory {"/content/fnl_model"}

    if download==True:
        !zip /content/fnl_model.zip /content/fnl_model
        files.download("/content/fnl_model.zip")


#------------------------------------------------------------------------------------------
#----------------------------------- LÍNEAS A MODIFICAR -----------------------------------
#------------------------------------------------------------------------------------------
download_model()
pipeline_config=conf_pipeline(32, 4500) 

train_model()

exporter_model(True)
#------------------------------------------------------------------------------------------

    '''
    f.write(text)
    f.close() 


def execution_colab():
    f = open ('consumo_modelo.txt','w')
    text='''
#-----------------------INSTALACION OBJECT_DETECTION_API-----------------------
#--------------(SOLO ES NECESARIO INSTALARLO 1 VEZ - En una celda aparte)--------------
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


 #Para este scrip es necesario modificar las líneas indicadas y cargar en la raíz ("/content") los archivos **`"fnl_model.zip", "centroidtracker.py", "trackableobject.py" y "label_map.pbtxt"`**

#---------------------DEPENDENCIAS---------------------
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow


import numpy as np
import imutils
import time
import dlib
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject


if not os.path.exists("/content/fnl_model"):
    !unzip -qo /content/fnl_model.zip     #Descomprime el modelo


#---------------------FUNCIONES---------------------
def process_image(image_path, max_boxes=20, threshold=0.8):
    """Función que consume una imagen y entrega la imagen con los bounding box, las clases y los score.
    
    Args:
        image_path: Recibe la ruta a la imagen que será consumida
        max_boxes: Recibe un int que indica el numero max de objetos a detectar
        threshold: Recibe un float que indica el numero mínimo que tiene que tener un score para ser visualizado"""

    try:
        if detect_fn==None: 
            pass
    except:
        detect_fn = tf.saved_model.load("/content/fnl_model/saved_model") 
        category_index = label_map_util.create_category_index_from_labelmap("/content/label_map.pbtxt")
        

    image_np = np.array(Image.open(image_path))
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

    prediction = detect_fn(input_tensor)
    num_detections = int(prediction.pop('num_detections'))
    detections = {key: value[0,:num_detections].numpy() for key, value in prediction.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        max_boxes_to_draw=max_boxes,
        min_score_thresh=threshold,
        use_normalized_coordinates = True
    )
    return cv2_imshow(image_np_with_detections)

def process_video(video_path, skip_fps, threshold, function=None, option_vis=None):
    """Función que consume el video y entrega un nuevo video con las predicciones y los contadores, dependiendo la funcionalidad 
    seleccionada.
    
    Parameters:
        video_path: Recibe la ruta al video que será consumido
        skip_fps: Número de fps que el modelo estará en stand-by para realizar nuevas predicciones
        threshold: Recibe un float que indica el numero mínimo que tiene que tener un score para ser visualizado
        function = (tracker, counter)
            tracker: La función realizará un seguimiento de los objetos que cruzan por el video
            counter: La función contabilizará el número de objetos pertenecientes a las clases del modelo que se muestren en el video
        option_vis = (id, label, centroid, only_centroid)
            id: Se visualizará el bounding box, el centroide y su respectivo ID
            label: Se visualizará el bounding box, el centroide, la clase y el score
            centroid: Se visualizará el centroide y su ID
            only_centroid: Se visualizará únicamente el centroide de los objetos
    Return:
        predicción de video""" 


    try:
        if detect_fn==None: 
            pass
    except:
        detect_fn = tf.saved_model.load("/content/fnl_model/saved_model") 
        category_index = label_map_util.create_category_index_from_labelmap("/content/label_map.pbtxt")


    vs = cv2.VideoCapture(video_path)
    writer = None
    W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ct = CentroidTracker(maxDisappeared= 20, maxDistance = 30)

    trackers = []
    trackableObjects = {}

    if function=="counter": counters={cat["id"]:0 for cat in list(category_index.values())}

    totalFrame = 0
    totalDown = 0
    totalUp = 0

    fps = FPS().start()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter("/content/video_test_out.mp4", fourcc, 20.0, (W, H), True)

    while True:

        ret, frame = vs.read()

        if frame is None:
            break
        
        rects = []
        attributes = []

        if totalFrame % skip_fps == 0:
            status = "Detecting"
            trackers = []
            image_np = np.array(frame)

            input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
            detections = detect_fn(input_tensor)

            detection_scores = np.array(detections["detection_scores"][0])
            detection_clean = [x for x in detection_scores if x >= threshold]
            
            for x in range(len(detection_clean)):
                idx = int(detections['detection_classes'][0][x])
                ymin, xmin, ymax, xmax = np.array(detections['detection_boxes'][0][x])
                classes = int(np.array(detections['detection_classes'][0][x]))
                score = detection_clean[x]
                attribute = [classes, score]
                box = [xmin, ymin, xmax, ymax] * np.array([W, H, W, H])

                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(frame, rect)

                trackers.append(tracker)
                attributes.append(attribute)

        else:
            for tracker in trackers:
                status = "Watching"
                tracker.update(frame)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

                if option_vis=="label" or option_vis=="id": cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                
                if bool(to.attributes)==False and bool(attributes):
                    to.attributes = attributes

                    if not to.counted:
                        if function=="counter":
                            counters[to.attributes[0][0]] += 1
                            to.counted = True

                if not to.counted:
                    if function=="tracker" or function==None:
                        if direction < 0 and centroid[1] < H//2:
                            totalUp += 1
                            to.counted = True
                        elif direction > 0 and centroid[1] > H//2:
                            totalDown += 1
                            to.counted = True
                        

            trackableObjects[objectID] = to

            if function=="tracker" or function==None: cv2.line(frame, (0, H//2), (W, H//2), (0,0,255), 2)
                            
            if bool(to.attributes):
                if option_vis=="id" or option_vis==None:
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0]-20, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,0,255), -1)
                if option_vis=="label":
                    text = "{0} at {1:.2f}%".format(category_index[(to.attributes[0][0])]['name'],(float(to.attributes[0][1])*100))
                    cv2.putText(frame, text, (centroid[0]-50, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,0,255), -1)
                if option_vis=="centroid":
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0]-20, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,0,255), -1)
                if option_vis=="only_centroid":
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,0,255), -1)        
    

        if function=="tracker" or function==None:
            info = [("Subiendo", totalUp), ("Bajando", totalDown), ("Estado", status)]
            cv2.rectangle(frame, (5, H - ((len(info)*20) + 20)), ( int(W*0.25) , H - 10), (0, 0, 0), -1)

        if function=="counter":
            info = [("Labels: ", {name:cont for (name,cont) in zip(list(map(lambda x: x["name"] , list(category_index.values()))),list(counters.values()))}), ("Estado", status)]
            cv2.rectangle(frame, (5, H - ((len(info)*20) + 20)), ((len(info[0][1])*125), H - 10), (0, 0, 0), -1)

        for (i, (k,v)) in enumerate(info):
            text = "{}: {}".format(k,v)
            cv2.putText(frame, text, (10, H - ((i*20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        writer.write(frame)
        totalFrame += 1
        fps.update()

    fps.stop()

    print("Tiempo completo {}".format(fps.elapsed()))
    print("Tiempo aproximado por frame {}".format(fps.fps()))

    writer.release()
    vs.release()

#------------------------------------------------------------------------------------------
#----------------------------------- LÍNEAS A MODIFICAR -----------------------------------
#------------------------------------------------------------------------------------------
process_image("/content/enfermo1.jpeg")
process_video("/content/test_video5.mp4", 30, 0.8, function="counter", option_vis="label")
#------------------------------------------------------------------------------------------
 
    '''
    f.write(text)
    f.close() 