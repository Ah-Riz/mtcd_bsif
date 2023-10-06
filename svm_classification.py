import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate 
from sklearn import svm

# from datetime import datetime

import csv

import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
from multiprocessing import Pool
 
def svm_classifier(setup_mutiprocessing):
    # kernel = {1:"linear",2:"rbf",3:"poly",4:"sigmoid"}
    # print(setup_mutiprocessing)
    # path_ekstraksi = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",folder_ekstraksi,file)
    dt = np.genfromtxt(setup_mutiprocessing[0], delimiter=",")
    dt = np.delete(dt,0,1)

    target = np.array(dt[:,0]).astype(int)
    features = np.array(dt[:,1:])

    features = StandardScaler().fit_transform(features)
    # print(features.shape)
    
    if setup_mutiprocessing[3] != 0:
        features_length = features.shape[1]
        pca = round(features_length * setup_mutiprocessing[3])
        pca = PCA(n_components=pca)
        pca.fit(features)
        features = pca.transform(features)
        # print(features.shape)


    # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    # start = datetime.now()

    clf = svm.SVC(kernel=setup_mutiprocessing[2], C=setup_mutiprocessing[1], decision_function_shape='ovr')
    # clf = svm.SVC(kernel=kernel[svm_kernel], C=svm_c, decision_function_shape='ovo')
    scoring={"acc":"accuracy", "f1_micro":"f1_micro"}
    cv=10
    scores = cross_validate(clf, features, target, cv=cv, scoring=scoring, return_estimator=True, error_score="raise")

    # end = datetime.now()
    # tot_time = (end-start).total_seconds()
    # print("%0.3f for avg accuracy with avg f1-score: %0.3f" % np.average(scores["test_acc"]), np.average(scores["test_f1_micro"])))
    # print(np.average(scores["test_acc"]), np.average(scores["test_f1_micro"]))
    file = str(setup_mutiprocessing[0]).split('\\')[-1]
    return file, setup_mutiprocessing[1], setup_mutiprocessing[2], setup_mutiprocessing[3],np.average(scores["test_acc"]), np.average(scores["test_f1_micro"])

def run(path, direktory_ekstraksi_fitur,file_save):
    path_direktory_ekstraksi_fitur = os.path.join(path,direktory_ekstraksi_fitur)
    c = [1,10,100]
    svm_kernel = ["linear","rbf","poly","sigmoid"]
    pca = [0.2, 0.4, 0.6, 0.8, 0]
    name_of_file =  np.array(sort(os.listdir(path_direktory_ekstraksi_fitur)),dtype=object)
    # res = [["file", "c", "tipe kernel", "pca %", "avg acc", "avg f1-score"]]
    setup_mutiprocessing = []
    for file in name_of_file:
        for j in range(len(c)):
            for kernel in svm_kernel:
                for i in range(len(pca)):
                    setup_mutiprocessing.append([os.path.join(path_direktory_ekstraksi_fitur,file), c[j], kernel, pca[i]])
    # start = datetime.now()
    with Pool() as pool:
        res = pool.map(svm_classifier,setup_mutiprocessing)
        pool.close()
    # end = datetime.now()
    # tot_cpu = (end-start).total_seconds()

                    # avg_acc, avg_f1 = svm_classifier(os.path.join(path_direktory_ekstraksi_fitur,file), c, kernel, pca[i])
                    # print("-",file,c,kernel,pca)
                    # res.append([file,c,kernel,pca,avg_acc, avg_f1])
    # print("cpu:",tot_cpu)

    csv_file = os.path.join(path,direktory_ekstraksi_fitur,file_save)
    if not os.path.isfile(csv_file):
      # If the CSV file doesn't exist, create it and write a header
      with open(csv_file, 'w', newline='') as file:
          writer = csv.writer(file)
    with open(csv_file,'a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["file", "c", "tipe kernel", "pca %", "avg acc", "avg f1-score"])
      for i in res:
        writer.writerow(i)
    

if __name__ == '__main__':
    direktory_ekstraksi_fitur = "20231006_213519"
    path = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur")
    file_save = "_evaluasi.csv"
    
    if not os.path.isfile(os.path.join(path,direktory_ekstraksi_fitur,file_save)):
        run(path,direktory_ekstraksi_fitur,file_save)
    else:
        print("evaluasi telah dilakukan sebelumnya")
                    
                
                 
    #svm_kernel=> 1.linier, 2.rbf, 3.polinomial, 4.sigmoid 
    # print("===fitur_mtcd===")
    # svm_classifier(folder_ekstraksi,"fitur_mtcd.csv", c, svm_kernel=1, pca=0)
    # svm_classifier(folder_ekstraksi,"fitur_mtcd.csv", c, svm_kernel=1, pca=0.3)
    # print("===bsif_3x3_5bit===")
    # svm_classifier(folder_ekstraksi,"bsif_3x3_5bit.csv", c, svm_kernel=1, pca=0)
    # svm_classifier(folder_ekstraksi,"bsif_3x3_5bit.csv", c, svm_kernel=1, pca=0.3)
    # print("===fusion_3x3_5bit===")
    # svm_classifier(folder_ekstraksi,"fusion_3x3_5bit.csv", c, svm_kernel=1, pca=0)
    # svm_classifier(folder_ekstraksi,"fusion_3x3_5bit.csv", c, svm_kernel=1, pca=0.3)
    # print("===bsif_17x17_12bit===")
    # svm_classifier(folder_ekstraksi,"bsif_17x17_12bit.csv", c, svm_kernel=1, pca=0)
    # svm_classifier(folder_ekstraksi,"bsif_17x17_12bit.csv", c, svm_kernel=1, pca=0.3)
    # print("===fusion_17x17_12bit===")
    # svm_classifier(folder_ekstraksi,"fusion_17x17_12bit.csv", c, svm_kernel=1, pca=0)
    # svm_classifier(folder_ekstraksi,"fusion_17x17_12bit.csv", c, svm_kernel=1, pca=0.3)