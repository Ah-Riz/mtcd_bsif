import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pylab import *
import csv
from datetime import datetime

from multiprocessing import Pool

import shutil

from MTCD import mtcd
from BSIF import bsif_gpu

def preprocessing(img_path, mode="mtcd"):
  image_name = img_path.split("\\")[-1]
  class_name = image_name.split(" ")[0]
  if mode == "bsif":
    img_grayscale = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    return img_grayscale, image_name, class_name
  else :
    img = cv2.imread(img_path)
    return img, image_name, class_name

def setup_bsif_gpu(img_path):
  img, image_name, class_name = preprocessing(img_path,"bsif")
  path_filter_BSIF = os.path.join("C:","\\Users","Rizki","Documents","thesis","texturefilters")
  all_filter_name =  np.array(sorted(os.listdir(path_filter_BSIF)),dtype=object)
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
      
  return result_bsif



def setup_mtcd(img_path):
  img, image_name, class_name = preprocessing(img_path,"mtcd")
  mtcd_result = mtcd(img)
  data_mtcd = np.concatenate(([image_name, class_name], mtcd_result))
  return data_mtcd

def save2csv(data, path_save, mode="mtcd"):
  if mode == "mtcd":
    csv_file = os.path.join(path_save,"fitur_mtcd.csv")
    if not os.path.isfile(csv_file):
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
    for i in range(len(bsif[key])):
      value_bsif = bsif[key][i][0][2:]
      res = np.concatenate((np.array(mtcd[i]), np.array(value_bsif)))
      if f_key in fusion_result.keys():
        fusion_result[f_key].append(res)
      else:
        fusion_result[f_key] = [res]
  return fusion_result

def augmentation(path_batik):
  path = os.path.normpath(path_batik + os.sep + os.pardir)
  all_batik_name =  np.array(sorted(os.listdir(path)),dtype=object)
  
  os.makedirs(path_batik)
  for i in range(len(all_batik_name)):
    shutil.copy2(os.path.join(path,all_batik_name[i]), path_batik)  
    src = cv2.imread(os.path.join(path,all_batik_name[i])) 
    img = cv2.flip(src, 0)
    name, extension = all_batik_name[i].split(".")
    name = name+"_flip."+extension
    cv2.imwrite(os.path.join(path_batik,name), img) 

if __name__ == '__main__':
  path_save = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur")
  path_batik = os.path.join("C:","\\Users","Rizki","Documents","thesis","Batik_Nitik_960_Images","augmented_image")
  

  if not os.path.exists(path_batik):
    augmentation(path_batik)
  
  all_batik_name =  np.array(sorted(os.listdir(path_batik)),dtype=object)
  
  for i in range(len(all_batik_name)):
    all_batik_name[i] = os.path.join(path_batik,all_batik_name[i])

  # Ekstraksi fitur MTCD multiprocessing
  with Pool() as pool:
    result_mtcd = pool.map(setup_mtcd, all_batik_name)
    pool.close()

  # Ekstraksi fitur BSIF multiprocessing
  gpu_bsif = []
  for i in divide_to_batch(all_batik_name,300):
    with Pool() as pool:
      result_bsif = pool.map(setup_bsif_gpu, i)
      gpu_bsif.append([result_bsif])
      pool.close()
  sorted_bsif_gpu = sort_bsif(gpu_bsif)
  
  # Penggabungan fitur
  fusion_result = fusion(result_mtcd, sorted_bsif_gpu)
  
  dt = str(datetime.now()).split(" ")
  date = dt[0].replace('-', '')
  time = dt[1].split(".")[0].replace(':','').replace('+','')
  target = date + "_" + time
  path_save = os.path.join(path_save, target)
  os.makedirs(path_save)
  
  save2csv(result_mtcd, path_save, 'mtcd')
  save2csv(sorted_bsif_gpu, path_save, 'bsif')
  save2csv(fusion_result, path_save, 'fusion')