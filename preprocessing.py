import utils
import json
import pickle
import pandas as pd


utils.rename_dataset("dataset")
utils.split_dataset("dataset",0.8)
utils.json_to_csv("labelStudio_train.json")
utils.json_to_csv("labelStudio_test.json")
