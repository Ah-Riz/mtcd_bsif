import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
from sklearn import svm

import pickle
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

import csv

import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
from multiprocessing import Pool

def custom_cv(data, k):
    if k==1:
        k_ = 1
        k = 2
    else:
        k_ = k
    kelas = {}

    for i in range(1, len(data)+1):
        if not data[i-1, 0] in kelas.keys():
            kelas[data[i-1, 0]] = np.array((data[i-1,:]))
        else:
            kelas[data[i-1, 0]] = np.vstack((kelas[data[i-1, 0]], data[i-1,:]))
    if k_ != 1:
        col_count = int(32/k)
    else:
        col_count = int(16/k)
    selec = np.reshape(np.array((range(kelas[1].shape[0]))), (-1, col_count))
    
    split = {}
    for i in range(k):
        for j in list(kelas.keys()):
            if not i in list(split.keys()):
                split[i] = np.array((kelas[j][selec[i,0]]))
                for l in range(1,col_count):
                    split[i] = np.vstack((split[i], kelas[j][selec[i,l]]))

            else:
                for l in range(col_count):
                    split[i] = np.vstack((split[i], kelas[j][selec[i,l]]))
    # if k_:
    #     k=1
    for i in range(k_):
        test = split[i]
        train = np.array(())            
        b = np.delete(np.arange(k), i)
        for j in b:
            if len(train) == 0:
                train = np.array((split[j]))
            else:
                train = np.vstack((train, split[j]))
    
        yield train, test

def custom_cross_validate_SVM(data, cv, kernel, C, decision_function_shape='ovr'):
    clf = svm.SVC(kernel=kernel, C=C, decision_function_shape=decision_function_shape)
    
    acc = []
    macrof1 = []
    real = []
    pred = []
    
    # if cv != 1:
    for train, test in custom_cv(data, cv):
        X_train = train[:,1:]
        y_train = train[:,0]
        X_test = test[:,1:]
        y_test = test[:,0]

        model = clf.fit(X_train, y_train)

        pre = model.predict(X_test)

        if str(np.array(acc).shape) == "(0,)":
            acc = np.array(clf.score(X_test, y_test))
            macrof1 = np.array(f1_score(y_test, pre, average='macro'))
            real = np.array(y_test)
            pred = np.array(pre)
        else:
            acc = np.vstack((acc, clf.score(X_test, y_test)))
            macrof1 = np.vstack((macrof1, f1_score(y_test, pre, average='macro')))
            real = np.vstack((real, y_test))
            pred = np.vstack((pred, pre))
    # else:
    #     # X_train, X_test, y_train, y_test = train_test_split(data[:,1:], data[:,:1], test_size=0.125, random_state=0, stratify=60)
    #     X = data[:,1:]
    #     y = data[:,:1].ravel()
    #     split=StratifiedShuffleSplit(n_splits=1, test_size=0.25)
    #     split.get_n_splits(X, y) 
    #     for train_index, test_index in split.split(X, y): 
    #         X_train, X_test = X[train_index], X[test_index] 
    #         y_train, y_test = y[train_index], y[test_index] 
    #         model = clf.fit(X_train, y_train)
    #     pre = model.predict(X_test)
    #     acc = np.array(clf.score(X_test, y_test))
    #     macrof1 = np.array(f1_score(y_test, pre, average='macro'))
    #     real = np.array(y_test)
    #     pred = np.array(pre)
        
    return acc, macrof1, real, pred, model

def svm_classifier(setup_mutiprocessing):
    if setup_mutiprocessing[-2] != 1:
        pca = setup_mutiprocessing[0].split("\\")[-2].split("_")[-1]
    else:
        pca = "-"
    dt = np.genfromtxt(setup_mutiprocessing[0], delimiter=",")
    dt = np.delete(dt,0,1)
    
    cv = setup_mutiprocessing[-2]
    
    acc, macrof1, real, pred, model = custom_cross_validate_SVM(dt, cv, setup_mutiprocessing[2], setup_mutiprocessing[1], "ovr")
    
    file = str(setup_mutiprocessing[0]).split('\\')[-1]
    if "mtcd" in file:
        metode = "mtcd"
        size = "-"
        bit = "-"
    else:
        metode, size, bit = str(setup_mutiprocessing[0]).split('\\')[-1].replace('bit.csv', '').split("_")
    
    data_cm = {"real":real, "pred":pred}
    
    if setup_mutiprocessing[-1] != 4:
        if not os.path.exists(os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",setup_mutiprocessing[0].split('\\')[-4],setup_mutiprocessing[0].split('\\')[-3],setup_mutiprocessing[0].split('\\')[-2],"pickel")):
            os.mkdir(os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",setup_mutiprocessing[0].split('\\')[-4],setup_mutiprocessing[0].split('\\')[-3],setup_mutiprocessing[0].split('\\')[-2],"pickel"))
        with open(os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",setup_mutiprocessing[0].split('\\')[-4],setup_mutiprocessing[0].split('\\')[-3],setup_mutiprocessing[0].split('\\')[-2],"pickel",f'_{metode}_{size}_{bit}bit_{setup_mutiprocessing[1]}_{setup_mutiprocessing[2]}_{pca}.pickle'), 'wb') as handle:
            pickle.dump(data_cm, handle)
        with open(os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",setup_mutiprocessing[0].split('\\')[-4],setup_mutiprocessing[0].split('\\')[-3],setup_mutiprocessing[0].split('\\')[-2],"pickel",f'MODEL_{metode}_{size}_{bit}bit_{setup_mutiprocessing[1]}_{setup_mutiprocessing[2]}_{pca}.pickle'), 'wb') as handle:
            pickle.dump(model, handle)
    else:
        if not os.path.exists(os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",setup_mutiprocessing[0].split('\\')[-3],setup_mutiprocessing[0].split('\\')[-2],"pickel")):
            os.mkdir(os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",setup_mutiprocessing[0].split('\\')[-3],setup_mutiprocessing[0].split('\\')[-2],"pickel"))
        with open(os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",setup_mutiprocessing[0].split('\\')[-3],setup_mutiprocessing[0].split('\\')[-2],"pickel",f'_{metode}_{size}_{bit}bit_{setup_mutiprocessing[1]}_{setup_mutiprocessing[2]}_{pca}.pickle'), 'wb') as handle:
            pickle.dump(data_cm, handle)
        with open(os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",setup_mutiprocessing[0].split('\\')[-3],setup_mutiprocessing[0].split('\\')[-2],"pickel",f'MODEL_{metode}_{size}_{bit}bit_{setup_mutiprocessing[1]}_{setup_mutiprocessing[2]}_{pca}.pickle'), 'wb') as handle:
            pickle.dump(model, handle)
        
        

    return metode, size, bit, setup_mutiprocessing[1], setup_mutiprocessing[2], pca, np.average(acc), np.average(macrof1)

def svm_setup(path_dir_fitur, file_save, cv, c, svm_kernel, preprocess_mode):
    name_of_file = [f for f in os.listdir(path_dir_fitur) if os.path.isfile(os.path.join(path_dir_fitur, f)) and f != file_save]
    
    # c = [1,10,100]
    # svm_kernel = ["linear","rbf","poly","sigmoid"]
        
    setup_mutiprocessing = []
    for file in name_of_file:
        for j in range(len(c)):
            for kernel in svm_kernel:
                setup_mutiprocessing.append([os.path.join(path_dir_fitur,file), c[j], kernel, cv, preprocess_mode])
        
    # res = [svm_classifier(setup_mutiprocessing[0])]
    with Pool() as pool:
        res = pool.map(svm_classifier,setup_mutiprocessing)
        
    csv_file = os.path.join(path_dir_fitur,file_save)
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["metode", "size", "bit", "c", "tipe kernel", "pca %", "avg acc", "avg_mac_f1"])
    with open(csv_file,'a', newline='') as file:
        writer = csv.writer(file)
        for i in res:
            writer.writerow(i)