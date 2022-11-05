"""---------------------------------------------------------------------------------------------------
Este script contiene las funciones:
    rename_dataset, 
    split_dataset, 
    json_to_csv, 

Funciones ejecutadas en google colaboratory:
    *create_labelmap, 
    *create_tfrecords,
    *conf_pipeline,
    *train_model,
    *exporter_model,
    *process_image,
    *process_video
---------------------------------------------------------------------------------------------------"""

import os
import shutil
import random
import json
import pandas as pd


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
            ruta=os.path.join(ruta, path)
            rename_dataset(ruta)
            ruta=os.path.split(ruta)[0]              
    else:                
        carpeta=os.path.split(os.path.split(ruta)[0])[1]
        imagen=carpeta+str(count)+path[path.find("."):]
        os.rename(ruta,os.path.join(os.path.split(ruta)[0],imagen))
    
                
def split_dataset(ruta, train):
    """Renombra la carpeta original del dataset y se creará una carpeta que contendrá el 
    dataset train y test en función del % introducido en la función split_dataset
    
    Args:
        ruta: Recibe la ruta a la carpeta que contiene el dataset 
        train= float: Porcentaje de archivos que contendrá la carpeta train
    Return:
        orig_dataset/
        fnl_dataset/"""
    try:
        shutil.copytree(ruta, "orig_dataset")
    except:
        pass

    if os.path.isdir(ruta):
        ruta=ruta
        for p in os.listdir(ruta):
            global path
            path=p
            ruta=os.path.join(ruta, path)
            split_dataset(ruta, train)
            ruta=os.path.split(ruta)[0]
    else:
        os.makedirs("fnl_dataset/train", exist_ok=True)
        os.makedirs("fnl_dataset/test", exist_ok=True)     
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