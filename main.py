import utils

#Empezamos renombrando y diviendo el dataset en train y test
utils.rename_dataset("dataset")
utils.split_dataset("dataset", 0.8)

#Luego de utilizar el labelStudio para etiquetar nuestro dataset, procedemos con la creación de los inputs del modelo
    #Aprovechando la capacidad de computo de google, el resto del código fué ejecutado en Colab!!!
utils.json_to_csv("labelStudio_train.json")
utils.json_to_csv("labelStudio_test.json")

utils.create_labelmap("/content/labelStudio_train.csv")
utils.create_tfrecords("/content/labelStudio_train.csv", "/content/labelStudio_test.csv", True)

#Una vez obtenidos el labelmap y los .records de entrenamiento y test, modificamos el pipeline y entrenamos el modelo
utils.conf_pipeline(32, 4500) 
utils.train_model()
utils.exporter_model(True)


#Testeamos el modelo tanto para imagenes como para videos
utils.process_image("/content/enfermo1.jpeg")
utils.process_video("/content/test_video5.mp4", 30, 0.8, function="counter", option_vis="label")