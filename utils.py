"""---------------------------------------------------------------------------------------------------
Este script contiene funciones prácticas/necesarias para la implementacion de modelos de deep learning 
en vision artificial.
---------------------------------------------------------------------------------------------------"""

import os
import shutil
import random
import json
import pandas as pd

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import io
import tensorflow as tf
import sys
sys.path.append("../../models/research")

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


def rename_dataset(ruta):
    '''Renombra los archivos del dataset, cada archivo recibirá el nombre de la carpeta que 
    lo contine y un número'''

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
    dataset train y test en función del % introducido en la función split_dataset"""

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
    para la implementación en modelos de tensorflow"""
    
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


def create_labelmap(csv):
    '''Crea el labelmap necesario en el procesamiento de los modelos de tensorflow'''

    if not os.path.exists("label_map.pbtxt"):
        label_dic = pd.DataFrame(pd.read_csv(csv))["class"].unique()
        label_dic = dict(enumerate(label_dic,start= 1))
        label_dic = dict(map(reversed, label_dic.items()))        
        with open("label_map.pbtxt", "w") as f:
            for keys, values in label_dic.items():
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(keys))
                f.write('\tid:{}\n'.format(values))
                f.write('}\n')
    else:
        item_id = None
        item_name = None
        label_dic = {}
        with open("label_map.pbtxt", "r") as file:
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


def class_text_to_int(row_label, labelmap):
    '''Funcion necesaria para creacion de los TFRecords, crea las clases del TFRecord basado en
    el labelmap'''

    if labelmap.get(row_label) != None:  
        return labelmap[row_label]
    else:
        None


def split(df, group):
    '''Funcion necesaria para creacion de los TFRecords, crea las clases del TFRecord basado en
    el labelmap'''

    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    '''Funcion que crea los TFRecords'''

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


def raise_tfrecords(ruta_train, ruta_test):
    '''Creación de los TFRecords de train y test a partir de las rutas de los .csv de train y
     test'''
    
    if os.path.exists("label_map.pbtxt"):
        output_path = "train.record"
        path = "fnl_dataset/train"
        
        global label_map
        label_map = create_labelmap(ruta_train)
        writer = tf.io.TFRecordWriter(output_path)
        examples = pd.read_csv(ruta_train)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))


        output_path = "test.record"
        path = "fnl_dataset/test"
        
        label_map = create_labelmap(ruta_test)
        writer = tf.io.TFRecordWriter(output_path)
        examples = pd.read_csv(ruta_test)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))

    else:
        create_labelmap(ruta_train)
        raise_tfrecords(ruta_train, ruta_test)