import os
import cv2
import numpy as np
import shutil

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
    
    return True

def grayscaling(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

def get_image(img_path, mode="mtcd"):
    image_name = img_path.split("\\")[-1]
    class_name = image_name.split(" ")[0]
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (256, 256))
    
    if mode == "bsif":
        img = grayscaling(img)
    
    return img, image_name, class_name