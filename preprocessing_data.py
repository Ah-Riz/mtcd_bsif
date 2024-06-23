import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import csv
import shutil

def save2csv(data, path_save):
    if not os.path.isfile(path_save):
        with open(path_save, 'w', newline='') as file:
            writer = csv.writer(file)
    with open(path_save, 'a', newline='') as file:
        writer= csv.writer(file)
        for i in data:
            writer.writerow(i)

def makedir(path):
    try:
        os.makedirs(path)
    except:
        pass

def scaling(data, path, name):
    scaler = MinMaxScaler(feature_range=(0,1))
    # scaler = StandardScaler()
    data[:,2:] = scaler.fit_transform(data[:, 2:])
    joblib.dump(scaler, os.path.join(path, f"{name}.joblib"))
    
    return data

def to_pca(data, length, path, name):
    
    if length != 1:
        n_comp = int((data.shape[1]-2) * length)
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(data[:, 2:])
        joblib.dump(pca, os.path.join(path, f"{name}.joblib"))
        data = np.hstack((data[:,:2], reduced))
    
    return data

def combination1(dir_f, dir_s, target, pca_l):
    bsif_file = [f for f in os.listdir(dir_f) if os.path.isfile(os.path.join(dir_f, f)) and f!="_evaluasi.csv" and f!="fitur_mtcd.csv"]
    
    # all_file.remove("fitur_mtcd.csv")
    
    scale_path = os.path.join(target, "scale")
    makedir(scale_path)
    scale_joblib_path = os.path.join(scale_path, "joblib")
    makedir(scale_joblib_path)
    
    pca_path = os.path.join(target, f"pca_{pca_l}")
    makedir(pca_path)
    pca_joblib_path = os.path.join(pca_path, "joblib")
    makedir(pca_joblib_path)
    
    #mtcd
    f_mtcd = np.genfromtxt(os.path.join(dir_f, "fitur_mtcd.csv"), delimiter=",")
    f_mtcd_s = np.genfromtxt(os.path.join(dir_s, "fitur_mtcd.csv"), delimiter=",")

    f_mtcd_sr = to_pca(f_mtcd_s, pca_l, pca_joblib_path, "fitur_mtcd.csv")
    csv_mtcd_sr = os.path.join(pca_path, "fitur_mtcd.csv")
    save2csv(f_mtcd_sr, csv_mtcd_sr)
    
    for i in range(len(bsif_file)):
        # bsif
        f_bsif = np.genfromtxt(os.path.join(dir_f, bsif_file[i]), delimiter=",")
        f_bsif_s = np.genfromtxt(os.path.join(dir_s, bsif_file[i]), delimiter=",")

        f_bsif_sr = to_pca(f_bsif_s, pca_l, pca_joblib_path, bsif_file[i])
        csv_bsif_sr = os.path.join(pca_path, bsif_file[i])
        save2csv(f_bsif_sr, csv_bsif_sr)
        
        # mtcd+bsif
        name = bsif_file[i].replace("bsif", "fusion")
        
        f_mb = np.hstack((f_mtcd, f_bsif[:, 2:]))
        f_mb_s = scaling(f_mb, scale_joblib_path, name)
        csv_mb_s = os.path.join(scale_path, name)
        save2csv(f_mb_s, csv_mb_s)

        f_mb_sr = to_pca(f_mb_s, pca_l, pca_joblib_path, name)
        csv_mb_sr = os.path.join(pca_path, name)
        save2csv(f_mb_sr, csv_mb_sr)
        
        # bsif+mtcd
        name = bsif_file[i].replace("bsif", "0fusion")
        
        f_bm = np.hstack((f_bsif, f_mtcd[:, 2:]))
        f_bm_s = scaling(f_bm, scale_joblib_path, name)
        csv_bm_s = os.path.join(scale_path, name)
        save2csv(f_bm_s, csv_bm_s)

        f_bm_sr = to_pca(f_bm_s, pca_l, pca_joblib_path, name)
        csv_bm_sr = os.path.join(pca_path, name)
        save2csv(f_bm_sr, csv_bm_sr)

def combination2(dir, target, pca_l):
    bsif_file = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f!="_evaluasi.csv" and f!="fitur_mtcd.csv"]
    
    # all_file.remove("fitur_mtcd.csv")
    
    scale_path = os.path.join(target, "scale")
    makedir(scale_path)
    scale_joblib_path = os.path.join(scale_path, "joblib")
    makedir(scale_joblib_path)
    
    pca_path = os.path.join(target, f"pca_{pca_l}")
    makedir(pca_path)
    pca_joblib_path = os.path.join(pca_path, "joblib")
    makedir(pca_joblib_path)
    
    # mtdc
    f_mtcd_s = np.genfromtxt(os.path.join(dir, "fitur_mtcd.csv"), delimiter=",")

    f_mtcd_sr = to_pca(f_mtcd_s, pca_l, pca_joblib_path, "fitur_mtcd.csv")
    csv_mtcd_sr = os.path.join(pca_path, "fitur_mtcd.csv")
    save2csv(f_mtcd_sr, csv_mtcd_sr)
    
    for i in range(len(bsif_file)):
        # bsif
        f_bsif_s = np.genfromtxt(os.path.join(dir, bsif_file[i]), delimiter=",")

        f_bsif_sr = to_pca(f_bsif_s, pca_l, pca_joblib_path, bsif_file[i])
        csv_bsif_sr = os.path.join(pca_path, bsif_file[i])
        save2csv(f_bsif_sr, csv_bsif_sr)
        
        # mtcd+bsif
        name = bsif_file[i].replace("bsif", "fusion")
        
        f_mb_s = np.hstack((f_mtcd_s, f_bsif_s[:, 2:]))
        csv_mb_s = os.path.join(scale_path, name)
        save2csv(f_mb_s, csv_mb_s)
        
        f_mb_sr = to_pca(f_mb_s, pca_l, pca_joblib_path, name)
        csv_mb_sr = os.path.join(pca_path, name)
        save2csv(f_mb_sr, csv_mb_sr)
        
        # bsif+mtcd
        name = bsif_file[i].replace("bsif", "0fusion")
        
        f_bm_s = np.hstack((f_bsif_s, f_mtcd_s[:, 2:]))
        csv_bm_s = os.path.join(scale_path, name)
        save2csv(f_bm_s, csv_bm_s)

        f_bm_sr = to_pca(f_bm_s, pca_l, pca_joblib_path, name)
        csv_bm_sr = os.path.join(pca_path, name)
        save2csv(f_bm_sr, csv_bm_sr)

def combination3(dir, target, pca_l):
    bsif_file = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f!="_evaluasi.csv" and f!="fitur_mtcd.csv"]

    # all_file.remove("fitur_mtcd.csv")

    scale_path = os.path.join(target, "scale")
    makedir(scale_path)
    scale_joblib_path = os.path.join(scale_path, "joblib")
    makedir(scale_joblib_path)

    pca_path = os.path.join(target, f"pca_{pca_l}")
    makedir(pca_path)
    pca_joblib_path = os.path.join(pca_path, "joblib")
    makedir(pca_joblib_path)

    #mtcd

    f_mtcd_s = np.genfromtxt(os.path.join(dir, "fitur_mtcd.csv"), delimiter=",")

    f_mtcd_sr = to_pca(f_mtcd_s, pca_l, pca_joblib_path, "fitur_mtcd.csv")
    csv_mtcd_sr = os.path.join(pca_path, "fitur_mtcd.csv")
    save2csv(f_mtcd_sr, csv_mtcd_sr)
    
    for i in range(len(bsif_file)):
        # bsif
        f_bsif_s = np.genfromtxt(os.path.join(dir, bsif_file[i]), delimiter=",")

        f_bsif_sr = to_pca(f_bsif_s, pca_l, pca_joblib_path, bsif_file[i])
        csv_bsif_sr = os.path.join(pca_path, bsif_file[i])
        save2csv(f_bsif_sr, csv_bsif_sr)
        
        # mtcd+bsif
        name = bsif_file[i].replace("bsif", "fusion")

        f_mb_sr = np.hstack((f_mtcd_sr, f_bsif_sr[:, 2:]))
        csv_mb_sr = os.path.join(pca_path, name)
        save2csv(f_mb_sr, csv_mb_sr)
        
        # bsif+mtcd
        name = bsif_file[i].replace("bsif", "0fusion")
        
        f_bm_sr = np.hstack((f_bsif_sr, f_mtcd_sr[:, 2:]))
        csv_bm_sr = os.path.join(pca_path, name)
        save2csv(f_bm_sr, csv_bm_sr)

def combination4(dir, target):
    all_file = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f!="_evaluasi.csv"]
    
    all_file.remove("fitur_mtcd.csv")
    
    f_mtcd = np.genfromtxt(os.path.join(dir, "fitur_mtcd.csv"), delimiter=",")
    
    for f in range(len(all_file)):
        shutil.copyfile(os.path.join(dir, all_file[f]), os.path.join(target, all_file[f]))
        
        f_bsif = np.genfromtxt(os.path.join(dir, all_file[f]), delimiter=",")
        
        # mtcd+bsif
        name = all_file[f].replace("bsif", "fusion")
        
        f_mb = np.hstack((f_mtcd, f_bsif[:, 2:]))
        csv_mb = os.path.join(target, name)
        save2csv(f_mb, csv_mb)
        
        # bsif+mtcd
        name = all_file[f].replace("bsif", "0fusion")
        
        f_bm = np.hstack((f_bsif, f_mtcd[:, 2:]))
        csv_bm = os.path.join(target, name)
        save2csv(f_bm, csv_bm) 

    shutil.copyfile(os.path.join(dir, "fitur_mtcd.csv"), os.path.join(target, "fitur_mtcd.csv"))
    

def preprocessing_setup(dir, mode=1, pca_l=1):
    path = os.path.join(dir, "feature")
    target = os.path.join(dir, f"combination{mode}")
    if mode != 4:
        scale_path = os.path.join(dir, "scale")
        if not os.path.exists(scale_path):
            makedir(scale_path)
            scale_joblib_path = os.path.join(scale_path, "joblib")
            makedir(scale_joblib_path)
            
            all_file = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f!="_evaluasi.csv"]
            for i in range(len(all_file)):
                f = np.genfromtxt(os.path.join(path, all_file[i]), delimiter=",")
                f_s = scaling(f, scale_joblib_path, all_file[i])
                csv_f_s = os.path.join(scale_path, all_file[i])
                save2csv(f_s, csv_f_s)

    if mode == 1:
        makedir(target)
        combination1(path, scale_path, target, pca_l)
    elif mode == 2:
        makedir(target)
        combination2(scale_path, target, pca_l)
    elif mode == 3:
        makedir(target)
        combination3(scale_path, target, pca_l)
    elif mode == 4:
        makedir(target)
        combination4(path, target)