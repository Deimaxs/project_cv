<!-- PROJECT LOGO -->
<br />

<p align="center">

![Logo WhiteFox](https://raw.githubusercontent.com/Deimaxs/project_cv/main/logo4.png)

  <h3 align="center">ProyectCV</h3>

  <p align="center">
    Objects tracker and counter on video
    <br />
    <br />
</p>



## Introducción

El objetivo del proyecto es realizar una API en la Google Cloud Platform (GCP), la cual recibe un video codificado en base64 y aplicará en él un modelo de seguimiento de direccion de objetos a traves de un umbral o un modelo de conteo de objetos.


* Entrada:
```json
{
  "video": "Video de entrada en base64",
  "skip": 30,
  "threshold": 0.8,
  "function": "tracker",
  "object_vis": "centroid"
}
```

* Salida:
```json
{
  "video": "Video de salida en base64",
}
```

Este script hace uso de herramientas como GCP y google colaboratory para la implementación de librerías y frameworks tales como tensorflow object-detection-API, OpenCV, Dlib entre otras.

<!-- USAGE EXAMPLES -->
## Procedimiento

**Modelo:**

* Ejecutar los metodos rename_dataset y split_datset, como se muestra en el archivo main.py

  ```python
  import utils

  utils.rename_dataset(path)
  utils.split_dataset(path, %split)
  ```

<!-- _For more examples, please refer to the [Examples packages](https://github.com/avmmodules/AVMWeather/tree/main/examples)_ -->


* Haciendo uso del labelStudio etiquetamos los sets de train y test, al finalizar descargamos los JSON Mini.

* Ejecutamos los metodos json_to_csv, como se muestra en el archivo main.py

  ```python
	utils.json_to_csv(path)
	utils.json_to_csv(path)
  ```

* Luego de utilizar el labelStudio para etiquetar nuestro dataset, procedemos con la creación de los inputs del modelo, para esto haremos uso de google colab provechando su potencia de computo.

* Utilizando el notebook process.ipynb en colab, ejecutamos las lineas en orden hasta llegar al apartado final donde se puede personalizar las funciones a gusto del usuario para testear y corroborar la eficiencia del modelo.

**Despliegue:**

* En VERTEX IA de GCP creamos un notebook, dentro del entrono generado copiamos la carpeta deploy.

* Dentro de la carpeta deploy copiamos el archivo.zip con nuestro modelo, el cual fué obtenido con la función exporter_model del colab y lo descomprimimos.

* Creamos un Artifact en GCP por medio del cloud shell.

  ```shell
  gcloud artifacts repositories create [NAME_FOLDER] --repository-format=docker --location=us-central1 --description="Docker repository"
  ```

* Generamos la imagen del contenedor para desplegar.

  ```shell
  gcloud builds submit --tag us-central1-docker.pkg.dev/[PROJECT_ID]/[NAME_FOLDER]/[NAME_IMAGE]:[NAME_TAG] --timeout=6000 
  ```
  
* Configuramos el entorno de Cloud Run para obtener la URL del contenedor desplegado y listo para recibir peticiones.

<!-- LICENSE -->
## License

  Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Email: mateo.sanchezalzate@gmail.com

Portfolio: 
[...](https://www.linkedin.com/in/mateo-sanchez-770019256/ "...")

LinkedIn: 
[Mateo Sánchez](https://www.linkedin.com/in/mateo-sanchez-770019256/ "Mateo Sánchez")

