# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

import cv2
# from google.colab.patches import cv2_imshow
import numpy as np

# %matplotlib inline
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pylab import *
# from skimage.feature import greycomatrix, greycoprops
#from skimage.feature import graycomatrix, graycoprops
#from sklearn.metrics.cluster import entropy

# import math

# import scipy

import csv
from datetime import datetime

from multiprocessing import Pool, Process, Condition, Event
from multiprocessing.dummy import Pool as ThreadPool

from time import sleep

from MTCD import *
from BSIF import *

def preprocessing(img_path, mode="mtcd"):
  image_name = img_path.split("\\")[-1]
  class_name = image_name.split(" ")[0]
  if mode == "bsif":
    img_grayscale = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    return img_grayscale, image_name, class_name
  else :
    img = cv2.imread(img_path)
    return img, image_name, class_name

# def setup_bsif(img_path, done, stop):
def setup_bsif(img_path):
  # with done:
    img, image_name, class_name = preprocessing(img_path,"bsif")
    # print(image_name)
    path_filter_BSIF = os.path.join("C:","\\Users","Rizki","Documents","thesis","texturefilters")
    all_filter_name =  np.array(sort(os.listdir(path_filter_BSIF)),dtype=object)
    result_bsif = {}
    for i in all_filter_name:
      filter_shape = i.split("_")[1]
      filter_bit = i.split("_")[-1].replace('bit.mat', '')
      bsif_result = bsif(img, os.path.join(path_filter_BSIF,i), i.split("_"))
      data_bsif = np.concatenate(([image_name, class_name],bsif_result))
      key = "bsif_"+filter_shape+"_"+filter_bit+"bit.csv"
      if key in result_bsif.keys():
        result_bsif[key].append(data_bsif)
      else:
        result_bsif[key] = [data_bsif]
    # print(result_bsif)
  #   done.notify()
    return result_bsif
  # while not stop.is_set():
  #   sleep(1)

def setup_bsif_gpu(img_path):
  # with done:
    img, image_name, class_name = preprocessing(img_path,"bsif")
    # print(image_name)
    path_filter_BSIF = os.path.join("C:","\\Users","Rizki","Documents","thesis","texturefilters")
    all_filter_name =  np.array(sort(os.listdir(path_filter_BSIF)),dtype=object)
    result_bsif = {}
    for i in all_filter_name:
      filter_shape = i.split("_")[1]
      filter_bit = i.split("_")[-1].replace('bit.mat', '')
      bsif_result = bsif_gpu(img, os.path.join(path_filter_BSIF,i), i.split("_"))
      data_bsif = np.concatenate(([image_name, class_name],bsif_result))
      key = "bsif_"+filter_shape+"_"+filter_bit+"bit.csv"
      if key in result_bsif.keys():
        result_bsif[key].append(data_bsif)
      else:
        result_bsif[key] = [data_bsif]
    # print(result_bsif)
  #   done.notify()
    return result_bsif
  # while not stop.is_set():
  #   sleep(1)



def setup_mtcd(img_path):
  img, image_name, class_name = preprocessing(img_path,"mtcd")
  # print(image_name)
  # ekstraksi mtcd
  mtcd_result = mtcd(img)
  data_mtcd = np.concatenate(([image_name, class_name], mtcd_result))
  return data_mtcd

def save2csv(data, path_save, mode="mtcd"):
  # if mode=="bsif":
    
  # else:
  if mode == "mtcd":
    csv_file = os.path.join(path_save,"fitur_mtcd.csv")
    if not os.path.isfile(csv_file):
      # If the CSV file doesn't exist, create it and write a header
      with open(csv_file, 'w', newline='') as file:
          writer = csv.writer(file)
    with open(csv_file,'a', newline='') as file:
      writer = csv.writer(file)
      for i in data:
        writer.writerow(i)
  elif mode == "fusion":
    bsif_key = list(data.keys())
    for key in bsif_key:
      csv_file = os.path.join(path_save,key)
      if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as file:
          writer = csv.writer(file)
      with open(csv_file, 'a', newline='') as file:
        writer= csv.writer(file)
        for i in data[key]:
          writer.writerow(i)
  elif mode == "bsif":
    bsif_key = list(data.keys())
    for key in bsif_key:
      csv_file = os.path.join(path_save,key)
      if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as file:
          writer = csv.writer(file)
      with open(csv_file, 'a', newline='') as file:
        writer= csv.writer(file)
        for i in data[key]:
          writer.writerow(i[0])
  elif mode == "bsif_gpu":
    bsif_key = list(data.keys())
    for key in bsif_key:
      csv_file = os.path.join(path_save,"gpu_"+key)
      if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as file:
          writer = csv.writer(file)
      with open(csv_file, 'a', newline='') as file:
        writer= csv.writer(file)
        for i in data[key]:
          writer.writerow(i[0])


def divide_to_batch(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]

def sort_bsif(all_bsif):
  sorted_bsif = {}
  for i in all_bsif:
    for j in i:
      for c in j:
        keyList = list(c.keys())
        for k in keyList:
          if k in sorted_bsif.keys():
            sorted_bsif[k].append(c[k])
          else:
            sorted_bsif[k] = [c[k]]
  return sorted_bsif

def fusion(mtcd, bsif):
  fusion_result = {}
  bsif_key = list(bsif.keys())
  for key in bsif_key:
    f_key = key.replace('bsif','fusion')
    print(np.array(bsif[key]).shape)
    for i in range(len(bsif[key])):
      value_bsif = bsif[key][i][0][2:]
      res = np.concatenate((np.array(mtcd[i]), np.array(value_bsif)))
      if f_key in fusion_result.keys():
        fusion_result[f_key].append(res)
      else:
        fusion_result[f_key] = [res]
  return fusion_result
        
      
  
  

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
  path_save = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur")
  path_batik = os.path.join("C:","\\Users","Rizki","Documents","thesis","Batik_Nitik_960_Images")
  
  dt = str(datetime.now()).split(" ")
  date = dt[0].replace('-', '')
  time = dt[1].split(".")[0].replace(':','').replace('+','')
  target = date + "_" + time
  path_save = os.path.join(path_save, target)
  os.makedirs(path_save)

  all_batik_name =  np.array(sort(os.listdir(path_batik)),dtype=object)

  for i in range(len(all_batik_name)):
    all_batik_name[i] = os.path.join(path_batik,all_batik_name[i])

  #ekstraksi fitur MTCD multiprocessing
  start = datetime.now()
  with Pool() as pool:
    result_mtcd = pool.map(setup_mtcd, all_batik_name)
    # save2csv(result_mtcd, path_save, "mtcd")
    pool.close()
  end = datetime.now()
  mtcd_time = (end-start).total_seconds()* 10**3

  # ekstraksi fitur BSIF multiprocessing
  all_bsif = []
  # a=0
  start = datetime.now()
  for i in divide_to_batch(all_batik_name,200):
    # a+=1
    with Pool() as pool:
      result_bsif = pool.map(setup_bsif, i)
      all_bsif.append([result_bsif])
      # save2csv(result_bsif, path_save, "bsif")
      pool.close()
    # if a==2:
    #   break
  sorted_bsif = sort_bsif(all_bsif)
  end = datetime.now()
  bsif_time = (end-start).total_seconds()* 10**3
  # print(sorted_bsif)

  fusion_result = fusion(result_mtcd, sorted_bsif)


  gpu_bsif = []
  # a=0
  start = datetime.now()
  for i in divide_to_batch(all_batik_name,200):
    # a+=1
    with Pool() as pool:
      result_bsif = pool.map(setup_bsif_gpu, i)
      gpu_bsif.append([result_bsif])
      pool.close()
  sorted_bsif_gpu = sort_bsif(gpu_bsif)
  end = datetime.now()
  bsif_gpu_time = (end-start).total_seconds()* 10**3
  
  save2csv(result_mtcd, path_save, 'mtcd')
  save2csv(sorted_bsif, path_save, 'bsif')
  save2csv(sorted_bsif_gpu, path_save, 'bsif_gpu')
  save2csv(fusion_result, path_save, 'fusion')
  
  
  # print(all_bsif)
  print(f"The time of execution of mtcd program is : {mtcd_time:.03f}ms")
  print(f"The time of execution of bsif program is : {bsif_time:.03f}ms")
  print(f"The time of execution of bsif gpu program is : {bsif_gpu_time:.03f}ms")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
