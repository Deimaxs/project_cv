import utils


#Empezamos renombrando y diviendo el dataset en train y test.
utils.rename_dataset("dataset")
utils.split_dataset("dataset", 0.8)

#Luego de utilizar el labelStudio para etiquetar nuestro dataset, procedemos con la creación de los inputs del modelo.
    #Aprovechando la capacidad de computo de google, el resto del código fué ejecutado en Colab!!!
utils.json_to_csv("labelStudio_train.json")
utils.json_to_csv("labelStudio_test.json")
