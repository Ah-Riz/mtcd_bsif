import os
import numpy as np
from multiprocessing import Pool
from datetime import datetime
import csv

from preprocesing_image import get_image
from MTCD import mtcd
from BSIF import bsif_gpu

def to_mtcd(img_path):
  img, image_name, class_name = get_image(img_path,"mtcd")
  mtcd_result = mtcd(img)
  data_mtcd = np.concatenate(([image_name, class_name], mtcd_result))
  
  return data_mtcd

def divide_to_batch(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def to_bsif_gpu(img_path):
    img, image_name, class_name = get_image(img_path,"bsif")
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

def feature_extraction(path_save, path_batik):
    all_batik_name = [f for f in os.listdir(path_batik) if os.path.isfile(os.path.join(path_batik, f))]
    for i in range(len(all_batik_name)):
        all_batik_name[i] = os.path.join(path_batik,all_batik_name[i])
    
    # Ekstraksi fitur MTCD multiprocessing
    with Pool() as pool:
        result_mtcd = pool.map(to_mtcd, all_batik_name)

    # Ekstraksi fitur BSIF multiprocessing
    gpu_bsif = []
    for i in divide_to_batch(all_batik_name,300):
        with Pool() as pool:
            result_bsif = pool.map(to_bsif_gpu, i)
            gpu_bsif.append([result_bsif])

    sorted_bsif_gpu = sort_bsif(gpu_bsif)

    dt = str(datetime.now()).split(" ")
    date = dt[0].replace('-', '')
    time = dt[1].split(".")[0].replace(':','').replace('+','')
    target = date + "_" + time

    path_dir = os.path.join(path_save, target)
    os.makedirs(path_dir)

    path_feature = os.path.join(path_save, target, "feature")
    os.makedirs(path_feature)

    save2csv(result_mtcd, path_feature, 'mtcd')
    save2csv(sorted_bsif_gpu, path_feature, 'bsif')

    return path_dir