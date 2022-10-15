import os
import shutil
import random
import json
import pandas as pd


def rename_dataset(ruta):
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
    data=json.load(open(ruta))
    csv_list=[]

    for img in data:
        width, height = img["label"][0]["original_width"], img["label"][0]["original_height"]
        image=img["image"]
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
