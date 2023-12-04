import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate 
from sklearn import svm

import csv

import matplotlib
matplotlib.use('Agg')
from matplotlib.pylab import *
from multiprocessing import Pool
 
def svm_classifier(setup_mutiprocessing):
    dt = np.genfromtxt(setup_mutiprocessing[0], delimiter=",")
    dt = np.delete(dt,0,1)
    
    target = np.array(dt[:,0]).astype(int)
    features = np.array(dt[:,1:])

    features = StandardScaler().fit_transform(features)
    
    if setup_mutiprocessing[3] != 1.0:
        features_length = features.shape[1]
        pca = round(features_length * setup_mutiprocessing[3])
        pca = PCA(n_components=pca)
        pca.fit(features)
        features = pca.transform(features)

    clf = svm.SVC(kernel=setup_mutiprocessing[2], C=setup_mutiprocessing[1], decision_function_shape='ovr')

    # scoring={"acc":"accuracy" , "precision_micro":"precision_micro", "recall_micro":"recall_micro","f1_micro":"f1_micro", "precision_macro":"precision_macro", "recall_macro":"recall_macro", "f1_macro":"f1_macro"}
    scoring={"acc":"accuracy", "f1_macro":"f1_macro"}
    cv=10
    scores = cross_validate(clf, features, target, cv=cv, scoring=scoring, return_estimator=True, error_score="raise")

    file = str(setup_mutiprocessing[0]).split('\\')[-1]
    return file, setup_mutiprocessing[1], setup_mutiprocessing[2], setup_mutiprocessing[3], np.average(scores["test_acc"]), np.average(scores["test_f1_macro"]), np.average(scores["fit_time"]), np.average(scores["score_time"])

def main(path, direktory_ekstraksi_fitur,file_save):
    path_direktory_ekstraksi_fitur = os.path.join(path,direktory_ekstraksi_fitur)
    c = [1,10,100]
    svm_kernel = ["linear","rbf","poly","sigmoid"]
    pca = [0.2, 0.4, 0.6, 0.8, 1.0]
    name_of_file =  np.array(sorted(os.listdir(path_direktory_ekstraksi_fitur)),dtype=object)

    setup_mutiprocessing = []
    for file in name_of_file:
        for j in range(len(c)):
            for kernel in svm_kernel:
                for i in range(len(pca)):
                    setup_mutiprocessing.append([os.path.join(path_direktory_ekstraksi_fitur,file), c[j], kernel, pca[i]])

    with Pool() as pool:
        res = pool.map(svm_classifier,setup_mutiprocessing)
        pool.close()

    csv_file = os.path.join(path,direktory_ekstraksi_fitur,file_save)
    if not os.path.isfile(csv_file):
      # If the CSV file doesn't exist, create it and write a header
      with open(csv_file, 'w', newline='') as file:
          writer = csv.writer(file)
    with open(csv_file,'a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["file", "c", "tipe kernel", "pca %", "avg acc", "avg f1-score macro", "avg fit time", "avg score time"])
      for i in res:
        writer.writerow(i)

if __name__ == '__main__':
    direktory_ekstraksi_fitur = "20231203_165222"
    path = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur")
    file_save = "_evaluasi.csv"
    
    if not os.path.isfile(os.path.join(path,direktory_ekstraksi_fitur,file_save)):
        main(path,direktory_ekstraksi_fitur,file_save)
    else:
        print("evaluasi telah dilakukan sebelumnya")