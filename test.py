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
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy

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
      bsif_result = bsif2(img, os.path.join(path_filter_BSIF,i), i.split("_"))
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
  # path_save = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur")

  # print(img_path)

  # img = cv2.imread(img_path)
  # img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # image_name = img_path.split("\\")[-1]
  # class_name = image_name.split(" ")[0]
  img, image_name, class_name = preprocessing(img_path,"mtcd")
  # print(image_name)
  # ekstraksi mtcd
  mtcd_result = mtcd(img)
  data_mtcd = np.concatenate(([image_name, class_name], mtcd_result))
  return data_mtcd
    # # ekstraksi bsif
    # path_kernel_BSIF = os.path.join("C:","\\Users","Rizki","Documents","thesis","texturefilters")
    # all_filter = sorted(os.listdir(path_kernel_BSIF))

    # for kname in all_filter:
    #     kname = kname.split("\\")[-1]
    #     kname_split = kname.split("_")
    
    #     kernel_shape = kname_split[1]
    #     # kernel_size = int(kernel_shape.split("x")[0])
    #     #
    #     bit = int(kname_split[-1].replace('bit.mat', ''))
    
    #     result_bsif = bsif(img_grayscale, path_kernel_BSIF, kname_split)
    #     # print(result_bsif)
    #     # print(len(result_bsif))
    
    #     data_bsif = np.concatenate(([image_name, class_name],result_bsif))
    #     target = "bsif_" + kernel_shape + "_" + str(bit) + "bit.csv"
    #     csv_file_bsif = os.path.join(path_save,target)
    
    #     if not os.path.isfile(csv_file_bsif):
    #         # If the CSV file doesn't exist, create it and write a header
    #         with open(csv_file_bsif, 'w', newline='') as file:
    #             writer = csv.writer(file)
    #     with open(csv_file_bsif,'a', newline='') as file:
    #         writer = csv.writer(file)
    #         if writer.writerow(data_bsif):
    #             file.close()




    # for i in range(len(all_filter)):
    #     all_filter[i] = os.path.join(path_kernel_BSIF,all_filter[i])
    # for filter in all_filter:
    #     result_bsif = bsif2(img_grayscale, path_kernel_BSIF)
    # with Pool() as pool:
    #     result = pool.map(bsif2, all_batik_name)

# def part_bsif()

# def run(path_batik, path_kernel_BSIF, path_save):
    # mtcd_feature = [[]]
    # bsif_feature = {}
    # jakarta_dt = str(datetime.now(tz=ZoneInfo("Asia/Jakarta"))).split(" ")
    # print(jakarta_dt)
    # dt = str(datetime.now()).split(" ")
    # date = dt[0].replace('-', '')
    # time = dt[1].split(".")[0].replace(':','').replace('+','')
    # target = str(date + "_" + time)
    # path_save = os.path.join(path_save, target)
    # # path_save = os.path.join(path_save, date + ' ' + time)
    # os.makedirs(path_save)
    # !mkdir path_save

  # all_batik_name = sort(os.listdir(path_batik))
  #   #   all_filter_name = os.listdir(path_kernel_BSIF)
  #   # print(all_batik_name)

  # for i in all_batik_name:
  #   i = os.path.join(path_batik,i)
  #   # print(all_batik_name)
  #   # process(i)
  # with Pool() as pool:
  #   result = pool.map(process, np.array(all_batik_name))

    # ======================================================

    # for name in sorted(os.listdir(path_batik)):
    # # return
    # print(name)
    # class_name = int(name.split(" ")[0])

    # img_path = os.path.join(path_batik,name)

    # img = cv2.imread(img_path)
    # # if resize!=512:
    # #   img_resize = cv2.resize(img, (resize, resize))
    # img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # mtcd_result = mtcd(img_resize)
    # mtcd_result = mtcd(img)
    # # print('aaa')
    # # print(mtcd_result.shape)
    # # print(mtcd_result[-1].shape)
    # # np.array(features)
    # # break
    # # data_mtcd = [class_name]
    # # data_mtcd.append(mtcd_result)
    # data_mtcd = np.concatenate(([name, class_name],mtcd_result))

    # # mtcd_feature.append(data_mtcd)
    # csv_file_mtcd = os.path.join(path_save,"fitur_mtcd.csv")
    # if not os.path.isfile(csv_file_mtcd):
    #     # If the CSV file doesn't exist, create it and write a header
    #     with open(csv_file_mtcd, 'w', newline='') as file:
    #         writer = csv.writer(file)
    # with open(csv_file_mtcd,'a', newline='') as file:
    #     writer = csv.writer(file)
    #     if writer.writerow(data_mtcd):
    #     file.close()
    #     # del mtcd_result, data_mtcd

    # for kname in sorted(os.listdir(path_kernel_BSIF)):
    #     # print(kname)
    #     kname_split = kname.split("_")
    
    #     kernel_shape = kname_split[1]
    #     # kernel_size = int(kernel_shape.split("x")[0])
    #   #
    #     bit = int(kname_split[-1].replace('bit.mat', ''))
    
    #     result_bsif = bsif(img_grayscale, path_kernel_BSIF, kname_split)
    #     # print(result_bsif)
    #     # print(len(result_bsif))
    
    #     data_bsif = np.concatenate(([name, class_name],result_bsif))
    #     target = "bsif_" + kernel_shape + "_" + str(bit) + "bit.csv"
    #     csv_file_bsif = os.path.join(path_save,target)
    
    #     if not os.path.isfile(csv_file_bsif):
    #     # If the CSV file doesn't exist, create it and write a header
    #     with open(csv_file_bsif, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #     with open(csv_file_bsif,'a', newline='') as file:
    #     writer = csv.writer(file)
    #     if writer.writerow(data_bsif):
    #         file.close()
    #         # del result_bsif, data_bsif

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
  # path_kernel_BSIF = os.path.join("C:","\\Users","Rizki","Documents","thesis","texturefilters")
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
  with Pool() as pool:
    result_mtcd = pool.map(setup_mtcd, all_batik_name)
    # save2csv(result_mtcd, path_save, "mtcd")
    pool.close()

  #ekstraksi fitur BSIF multiprocessing
  all_bsif = []
  # a=0
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
  # print(sorted_bsif)

  fusion_result = fusion(result_mtcd, sorted_bsif)

  save2csv(result_mtcd, path_save, 'mtcd')
  save2csv(sorted_bsif, path_save, 'bsif')
  save2csv(fusion_result, path_save, 'fusion')


  
    
  # print(fusion)
  #   keyList = list(data.keys())
  #   for i in keyList:

  #     file_name = i.replace('bsif','fusion')
  #     # csv_file = os.path.join(path_save,file_name)
  #     fusion[file_name] = 

    # done = Condition()
    # stop = Event()
    # result_bsif = Process(setup_bsif, args=(all_batik_name, done, stop))
    # with done:
    #   result_bsif.start()
    #   done.wait()
    # stop.set()
    # result_bsif.join()
    # print(result_bsif)

    # print(result_bsif)
    # csv_file_mtcd = os.path.join(path_save,"fitur_mtcd.csv")
    # if not os.path.isfile(csv_file_mtcd):
    #   # If the CSV file doesn't exist, create it and write a header
    #   with open(csv_file_mtcd, 'w', newline='') as file:
    #       writer = csv.writer(file)
    # with open(csv_file_mtcd,'a', newline='') as file:
    #   writer = csv.writer(file)
    #   for i in result_mtcd:
    #     writer.writerow(i)
      # if writer.writerow(result_mtcd):
        # file.close()
  # for i in all_batik_name:
  #   setup_bsif(i)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
