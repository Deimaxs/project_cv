import os

def rename_dataset(ruta):
    if os.path.isdir(ruta):
        for c,p in enumerate(os.listdir(ruta)):            
            global count,path
            count=c
            path=p
            ruta=os.path.join(ruta,path)
            rename_dataset(ruta)
            ruta=ruta[:-(len(p)+1)]              
    else:                
        carpeta=os.path.split(ruta[:-(len(path)+1)])[1]
        imagen=carpeta+str(count)+path[path.find("."):]
        print("Convertimos: {} en: {}".format(ruta,os.path.join(ruta[:-(len(path)+1)],imagen)))
        os.rename(ruta,os.path.join(ruta[:-(len(path)+1)],imagen))
                







    