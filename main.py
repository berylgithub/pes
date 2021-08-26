# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:35:09 2021

@author: Saint8312
"""
import json
from os import walk

data = {"H2":{"data":[]}, 
        "H3":{"data":[]}, 
        "H4":{"data":[]}, 
        "H5":{"data":[]}
        }

kinds = list(data.keys())
#read all data:
#f_dir = "C:/Users/beryl/Documents/WU/Main Topic/PythonFilesImportant/Data/Files/" # main pc dir
f_dir = "/users/baribowo/Documents/maintopic/PythonFilesImportant/Data/Files/" #office dir
_, _, onlyfiles = next(walk(f_dir))
print(onlyfiles)
for f in onlyfiles:
    f = open(f_dir+f)
    f_data = json.load(f)
    #data["H2"]["data"]
    print(f_data.keys(), len(f_data["data"]), f_data["kind"])
    data[f_data["kind"]]["data"].extend(f_data["data"])
    
