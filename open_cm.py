import pickle
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import numpy as np

def read_from_pickle2(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def main(dir,file):
    path = os.path.join(dir, file)
    
    data = read_from_pickle2(path)
    print(data)
    print(data["real"].shape)
    print(len(data["real"]))

    f, axes = plt.subplots(4, 4, figsize=(200, 200), sharey='row')
    f1 = []
    for i in range(len(data["real"])):
        f1.append(f1_score(data["real"][i], data["pred"][i], average='macro'))
        cf_matrix = confusion_matrix(data["real"][i], data["pred"][i])
        disp = ConfusionMatrixDisplay(cf_matrix)
        if i in [0,1,2,3]:
            x = 0
            y = i
        elif i in [4,5,6,7]:
            x = 1
            y = i-4
        elif i in [8,9,10,11]:
            x = 2
            y = i-4*2
        elif i in [12,13,14,15]:
            x = 3
            y = i-4*3
        disp.plot(ax=axes[x,y])
        disp.ax_.set_title(i)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i!=0:
            disp.ax_.set_ylabel('')
    
    f.text(0.4, 0.1, f'Predicted label f1={np.average(f1)}', ha='center')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    
    f.colorbar(disp.im_, ax=axes)
    if not os.path.exists(os.path.join(dir,"gambar")):
        os.mkdir(os.path.join(dir,"gambar"))
    plt.savefig(os.path.join(dir,"gambar",file+".png"))

if __name__ == "__main__":
    file = "_0fusion_3x3_5bit_1_linear_0.2"+".pickle"
    dir = "fin 2"
    path = os.path.join("C:","\\Users","Rizki","Documents","thesis","hasil_ekstraksi_fitur",dir,"pickel")
    main(path,file)