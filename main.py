import os
import numpy as np
from multiprocessing import Pool
from datetime import datetime

from preprocesing_image import augmentation
from feature_extract import feature_extraction
from preprocessing_data import preprocessing_setup
from classification import svm_setup

def main(aug = 1, dir=None, preprocess_mode = 1, pca_l = 0.2, cv=8, c = [1], svm_kernel=["linear"]):
    if cv <= 32 & aug == 1:
        cv=4
    elif cv <=16 & aug == 0:
        cv=4

    path_save = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur")
    # path_batik = os.path.join("C:","\\Users","Rizki","Documents","thesis","Batik_Nitik_960_Images","augmented_image")
    path_batik = os.path.join("C:","\\Users","Rizki","Documents","thesis","Batik_Nitik_960_Images")
    
    if aug == 1:
        if not os.path.exists(os.path.join(path_batik, "augmented_image")):
            augmentation(path_batik)
            path_batik = os.path.join(path_batik, "augmented_image")
        else:
            path_batik = os.path.join(path_batik, "augmented_image")

    if dir is None:
        dir = feature_extraction(path_save, path_batik)
    
    if preprocess_mode !=4:
        feature_path = os.path.join(dir, f"combination{preprocess_mode}", f"pca_{pca_l}")
    else:
        feature_path = os.path.join(dir, f"combination{preprocess_mode}")
    if not os.path.exists(feature_path):
        preprocessing_setup(dir, mode=preprocess_mode, pca_l=pca_l)
    
    file_save = "_evaluasi.csv"
    svm_setup(feature_path, file_save, cv=cv, c=c, svm_kernel=svm_kernel, preprocess_mode=preprocess_mode)
    
    return dir

if __name__ == '__main__':
    path = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur","aug_cv=8,mm")
    c = [1,10,100]
    svm_kernel = ["linear","rbf","poly","sigmoid"]
    cv=8
    for i in [1, 2, 3, 4]:
        for j in [1, 0.2, 0.4, 0.6, 0.8]:
            main(dir=path, preprocess_mode=i, pca_l=j, cv=cv, c=c, svm_kernel=svm_kernel)
            if i == 4:
                break
    
    path = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur","aug_cv=1,mm")
    c = [1]
    svm_kernel = ["linear"]
    cv=1
    for i in [1, 2, 3, 4]:
        for j in [1, 0.2, 0.4, 0.6, 0.8]:
            main(dir=path, preprocess_mode=i, pca_l=j, cv=cv, c=c, svm_kernel=svm_kernel)
            if i == 4:
                break