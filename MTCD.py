import numpy as np
import cv2
import math
from skimage.feature import graycomatrix, graycoprops


def colorQuantization(colorChannelImage, bins, max_color=255):
  #ganti varian warna ke bins varian
  quant = np.array(colorChannelImage) * (bins / max_color)
  quant = np.floor(quant)
  quant[quant >= bins] = bins -1

  return quant

def combineColorQuantization(image, B_bins, G_bins, R_bins):
  #ekstrak channel warna
  B = image[:,:,0]
  G = image[:,:,1]
  R = image[:,:,2]

  #kuantisasi tiap channel warna
  B_quant = colorQuantization(B, B_bins)
  G_quant = colorQuantization(G, G_bins)
  R_quant = colorQuantization(R, R_bins)

  #penggabungan kuantisasi warna
  combine_quant = (B_bins * G_bins * R_quant) + (B_bins * G_quant) + B_quant

  return combine_quant

def edgeQuantization(image, binsTheta):
  #ekstrak channel warna
  B = image[:,:,0]
  G = image[:,:,1]
  R = image[:,:,2]

  #sobel
  Bx = cv2.Sobel(B,cv2.CV_64F,1,0,ksize=3)
  By = cv2.Sobel(B,cv2.CV_64F,0,1,ksize=3)
  Gx = cv2.Sobel(G,cv2.CV_64F,1,0,ksize=3)
  Gy = cv2.Sobel(G,cv2.CV_64F,0,1,ksize=3)
  Rx = cv2.Sobel(R,cv2.CV_64F,1,0,ksize=3)
  Ry = cv2.Sobel(R,cv2.CV_64F,0,1,ksize=3)

  #|a| dan |b|
  a = np.sqrt(Bx**2 + Gx**2 + Rx**2)
  b = np.sqrt(By**2 + Gy**2 + Ry**2)

  #ab
  ab = (Bx*By) + (Gx*Gy) + (Rx*Ry)

  #orientasi gambar
  (h, w) = a.shape

  theta = np.zeros((h, w))

  for i in range(0, h):
    for j in range(0, w):
      if (a[i,j] == 0 or b[i,j] == 0):
        cosab1 = 0
      else:
        cosab1 = ab[i,j]/(a[i,j]*b[i,j])
      theta1 = math.degrees(np.arccos(cosab1))
      if (math.isnan(theta1)):
        theta1 = 0
      theta[i,j] = math.floor(theta1 * (binsTheta/180))
      if (theta[i,j] >= binsTheta-1):
        theta[i,j] = binsTheta-1

  return np.array(theta)

def textonSearch(colorQuant, cBins, edgeQuant, eBins):
  # define the shape of image
  (h, w) = colorQuant.shape

  # untuk menangani image yang dimensinya ganjil
  if (h % 2 == 1):
    a = np.zeros((h + 1, w))
    colorQuant = colorQuant[:-1, :]
    edgeQuant = edgeQuant[:-1, :]

  if (w % 2 == 1):
    a = np.zeros((h, w + 1))
    colorQuant = colorQuant[:, :-1]
    edgeQuant = edgeQuant[:, :-1]

  (h, w) = colorQuant.shape
  color_img = np.zeros((h, w))
  edge_img = np.zeros((h, w))

  # sliding window check for all color channel
  for i in range(0, h, 2):
    for j in range(0, w, 2):

      # texton search for color
      cTemp = colorQuant[i:i + 2, j:j + 2]
      if (cTemp[0, 0] == cTemp[0, 1]):  # texton type 1
        color_img[i, j] = colorQuant[i, j]
        color_img[i, j + 1] = colorQuant[i, j + 1]
      if (cTemp[0, 0] == cTemp[1, 0]):  # texton type 2
        color_img[i, j] = colorQuant[i, j]
        color_img[i + 1, j] = colorQuant[i + 1, j]
      if (cTemp[0, 0] == cTemp[1, 1]):  # texton type 3
        color_img[i, j] = colorQuant[i, j]
        color_img[i + 1, j + 1] = colorQuant[i + 1, j + 1]
      if (cTemp[1, 0] == cTemp[1, 1]):  # texton type 4
        color_img[i + 1, j] = colorQuant[i + 1, j]
        color_img[i + 1, j + 1] = colorQuant[i + 1, j + 1]
      if (cTemp[0, 1] == cTemp[1, 1]):  # texton type 5
        color_img[i, j + 1] = colorQuant[i, j + 1]
        color_img[i + 1, j + 1] = colorQuant[i + 1, j + 1]
      if (cTemp[0, 1] == cTemp[1, 0]):  # texton type 6
        color_img[i, j + 1] = colorQuant[i, j + 1]
        color_img[i + 1, j] = colorQuant[i + 1, j]

      # texton search for edge
      eTemp = edgeQuant[i:i + 2, j:j + 2]
      if (eTemp[0, 0] == eTemp[0, 1]):  # texton type 1
        edge_img[i, j] = edgeQuant[i, j]
        edge_img[i, j + 1] = edgeQuant[i, j + 1]
      if (eTemp[0, 0] == eTemp[1, 0]):  # texton type 2
        edge_img[i, j] = edgeQuant[i, j]
        edge_img[i + 1, j] = edgeQuant[i + 1, j]
      if (eTemp[0, 0] == eTemp[1, 1]):  # texton type 3
        edge_img[i, j] = edgeQuant[i, j]
        edge_img[i + 1, j + 1] = edgeQuant[i + 1, j + 1]
      if (eTemp[1, 0] == eTemp[1, 1]):  # texton type 4
        edge_img[i + 1, j] = edgeQuant[i + 1, j]
        edge_img[i + 1, j + 1] = edgeQuant[i + 1, j + 1]
      if (eTemp[0, 1] == eTemp[1, 1]):  # texton type 5
        edge_img[i, j + 1] = edgeQuant[i, j + 1]
        edge_img[i + 1, j + 1] = edgeQuant[i + 1, j + 1]
      if (eTemp[0, 1] == eTemp[1, 0]):  # texton type 6
        edge_img[i, j + 1] = edgeQuant[i, j + 1]
        edge_img[i + 1, j] = edgeQuant[i + 1, j]

  # make color histogram
  cF = np.histogram(color_img.ravel(), cBins, [0, 64])
  colorFeatures = (np.array(cF[0]) / 6)  # perlu dibagi dg 6 meyesuaikan dg jumlah type texton yg digunakan

  # make edge histogram
  eF = np.histogram(edge_img.ravel(), eBins, [0, 18])
  edgeFeatures = (np.array(eF[0]) / 6)

  # combine color and edge features
  features = []
  features.extend(colorFeatures)
  features.extend(edgeFeatures)

  return features

def GLCM(image):
  # convert iamge to greyscale
  grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # make co-occurance matrix
  gm = graycomatrix(grey_img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, normed=True)
  (h, w) = gm[:, :, 0, 0].shape

  glcm_features = []

  # calculate energy, contrast, correlation, entropy
  # using scikit library

  energy = graycoprops(gm, 'energy')
  contrast = graycoprops(gm, 'contrast')
  correlation = graycoprops(gm, 'correlation')
  glcm_features.extend(energy.tolist())
  glcm_features.extend(contrast.tolist())
  glcm_features.extend(correlation.tolist())

  entropy = []
  for i in range(0, 4):
    e = np.abs(gm[:, :, 0, i] * np.log2(gm[:, :, 0, i]))
    # print(e.shape)
    # print('eeeee')
    # print(type(e))
    # for i in range(e.shape[0]):
    #   for j in range(e.shape[1]):
    #     if math.isnan(e[i,j]):
    #       e[i, j] = 0
    e[np.isnan(e)] = 0
    entropy.append(np.sum(e))
    # print('-----------')
    # print(entropy)
    # print('-----------')
    # break

  return glcm_features, entropy

def mtcd(image):
  # Describe path of data train and index file
  # datatrain_path = root_path
  # indexfile_path = "BatikIndexing.csv"

  # define the bins for quantization
  colorBins = 64
  R_Bins = 4
  G_Bins = 4
  B_Bins = 4
  edgeBins = 18

  # open the output index file for writing
  # output = open(indexfile_path, 'w')

  # use glob to grab the image paths and logo over them
  # for imagePath in glob.glob(datatrain_path+'/*.jpg'):
  # extract the image ID (i.e. the unique filename) from the image path and load the image itself
  # imageID = imagePath[imagePath.rfind('/') + 1:]
  # image = cv2.imread(root_path)

  # Color Quantization
  colorQuant = combineColorQuantization(image, B_Bins, G_Bins, R_Bins)

  # Edge Quantization
  edgeQuant = edgeQuantization(image, edgeBins)

  # Texton Search
  # features = []
  features = textonSearch(colorQuant, colorBins, edgeQuant, edgeBins)

  # GLCM
  glcm, en = GLCM(image)

# write the features to file
  # features.extend(glcm)
  features.extend(glcm[0])
  features.extend(glcm[1])
  features.extend(glcm[2])
  features.extend(en)
  # print("===")
  # print(glcm[0].shape)
  # print(glcm[1].shape)
  # print(glcm[2].shape)
  # print("===")
  # print("===")
  # print(np.array(glcm).shape)
  # print(np.array(en).shape)
  # print("===")
  # features = np.array(features)
  # features = [str(f) for f in features]
  return np.array(features,dtype=object)
