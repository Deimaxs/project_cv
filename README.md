<!-- PROJECT LOGO -->
<br />

<p align="center">

![Logo WhiteFox](https://raw.githubusercontent.com/Deimaxs/project_cv/main/logo4.png)

  <h3 align="center">ProyectCV</h3>

  <p align="center">
    Objects tracker and counter on video (Using Dyno.py)
    <br />
    <br />
</p>



## Introducción

El objetivo del proyecto es realizar una API en la Google Cloud Platform, la cual recibe un video codificado en base64 y aplicará en él ya sea un modelo de trackeo de objetos en umbral, o un modelo de conteo de objetos.


* Entrada
```json
{
  "video": "Video de entrada en base64",
  "skip": 30,
  "threshold": 0.8,
  "function": "tracker",
  "object_vis": "centroid"
}
```

* Salida
```json
{
  "video": "Video de salida en base64",
}
```



<!-- USAGE EXAMPLES -->
## Procedimiento

* Ejecutar los metodos rename_dataset y split_datset, como se muestra en el archivo main.py

  ```python
  import utils

  #Empezamos renombrando y diviendo el dataset en train y test.
  utils.rename_dataset("dataset")
  utils.split_dataset("dataset", 0.8)
  ```

<!-- _For more examples, please refer to the [Examples packages](https://github.com/avmmodules/AVMWeather/tree/main/examples)_ -->


* Haciendo uso del labelStudio etiquetamos los sets de train y test, al finalizar descargamos los JSON Mini.

* Ejecutamos los metodos json_to_csv, como se muestra en el archivo main.py

* Luego de utilizar el labelStudio para etiquetar nuestro dataset, procedemos con la creación de los inputs del modelo.

  _Aprovechando la capacidad de computo de google, el resto del código fué ejecutado en Colab!!!_

  ```python
	utils.json_to_csv("labelStudio_train.json")
	utils.json_to_csv("labelStudio_test.json")
  ```

* Luego haciendo uso del process.ipynb en colab, ejecutamos las lineas en orden hasta llegar al apartado final donde se puede personalizar las funciones a gusto del usuario para testear y corroborar la eficiencia del modelo.

* En VERTEX IA de gcp creamos un notebook, dentro del entrono generado copiamos la carpeta deploy.

* Dentro de la carpeta deploy copiamos el archivo.zip con nuestro modelo obtenido con la función exporter_model del colab y lo descomprimimos.

* Creamos un artifact en gcp.

  ```python
  gcloud artifacts repositories create [NAME_FOLDER] --repository-format=docker --location=us-central1 --description="Docker repository"
  ```

* Haciendo uso de la carpeta deploy y de la plataforma GCP, generamos la imagen del contenedor para desplegar.

  ```python
  gcloud builds submit --tag us-central1-docker.pkg.dev/[PROJECT_ID]/[NAME_FOLDER]/[NAME_IMAGE]:[NAME_TAG] --timeout=6000 
  ```

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

