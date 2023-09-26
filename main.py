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

# from multiprocessing import Pool
# from multiprocessing.dummy import Pool as ThreadPool

from MTCD import *
from BSIF import *

def run(path_batik, path_kernel_BSIF, path_save):
  # mtcd_feature = [[]]
  # bsif_feature = {}
  # jakarta_dt = str(datetime.now(tz=ZoneInfo("Asia/Jakarta"))).split(" ")
  # print(jakarta_dt)
  dt = str(datetime.now()).split(" ")
  date = dt[0].replace('-', '')
  time = dt[1].split(".")[0].replace(':','').replace('+','')
  target = date + "_" + time
  path_save = os.path.join(path_save, target)
  # path_save = os.path.join(path_save, date + ' ' + time)
  os.makedirs(path_save)
  # !mkdir path_save

  for name in sorted(os.listdir(path_batik)):
    # return
    print(name)
    class_name = int(name.split(" ")[0])

    img_path = os.path.join(path_batik,name)

    img = cv2.imread(img_path)
    # if resize!=512:
    #   img_resize = cv2.resize(img, (resize, resize))
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # mtcd_result = mtcd(img_resize)
    mtcd_result = mtcd(img)
    # print('aaa')
    # print(mtcd_result.shape)
    # print(mtcd_result[-1].shape)
    # np.array(features)
    # break
    # data_mtcd = [class_name]
    # data_mtcd.append(mtcd_result)
    data_mtcd = np.concatenate(([name, class_name],mtcd_result))

    # mtcd_feature.append(data_mtcd)
    csv_file_mtcd = os.path.join(path_save,"fitur_mtcd.csv")
    if not os.path.isfile(csv_file_mtcd):
      # If the CSV file doesn't exist, create it and write a header
      with open(csv_file_mtcd, 'w', newline='') as file:
          writer = csv.writer(file)
    with open(csv_file_mtcd,'a', newline='') as file:
      writer = csv.writer(file)
      if writer.writerow(data_mtcd):
        file.close()
        # del mtcd_result, data_mtcd

    for kname in sorted(os.listdir(path_kernel_BSIF)):
      # print(kname)
      kname_split = kname.split("_")
    
      kernel_shape = kname_split[1]
      # kernel_size = int(kernel_shape.split("x")[0])
      #
      bit = int(kname_split[-1].replace('bit.mat', ''))
    
      result_bsif = bsif(img_grayscale, path_kernel_BSIF, kname_split)
      # print(result_bsif)
      # print(len(result_bsif))
    
      data_bsif = np.concatenate(([name, class_name],result_bsif))
      target = "bsif_" + kernel_shape + "_" + str(bit) + "bit.csv"
      csv_file_bsif = os.path.join(path_save,target)
    
      if not os.path.isfile(csv_file_bsif):
        # If the CSV file doesn't exist, create it and write a header
        with open(csv_file_bsif, 'w', newline='') as file:
            writer = csv.writer(file)
      with open(csv_file_bsif,'a', newline='') as file:
        writer = csv.writer(file)
        if writer.writerow(data_bsif):
          file.close()
          # del result_bsif, data_bsif

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
  path_batik = os.path.join("C:","\\Users","Rizki","Documents","thesis","Batik_Nitik_960_Images")
  path_kernel_BSIF = os.path.join("C:","\\Users","Rizki","Documents","thesis","texturefilters")
  path_save = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur")
  run(path_batik, path_kernel_BSIF, path_save)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
