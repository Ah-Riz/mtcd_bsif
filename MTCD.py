import numpy as np
import cv2
import math
from skimage.feature import graycomatrix, graycoprops


def colorQuantization(colorChannelImage, bins, max_color= 255):
  """
  Quantizes color values in the input image.

  Parameters:
  - colorChannelImage (np.ndarray): Input image array.
  - bins (int): Number of quantization bins.
  - max_color (int): Maximum color value in the input image. Default is 255.

  Returns:
  - np.ndarray: Quantized color values.
  """
  # Convert color values to quantized bins
  quant = np.array(colorChannelImage) * (bins / max_color)
  quant = np.floor(quant)
  quant[quant >= bins] = bins -1

  return quant

def combineColorQuantization(image, B_bins, G_bins, R_bins):
  """
  Combine quantized values from three color channels into a single representation.

  Parameters:
  - image (np.ndarray): Input image array.
  - B_bins (int): Number of bins for the blue channel.
  - G_bins (int): Number of bins for the green channel.
  - R_bins (int): Number of bins for the red channel.

  Returns:
  - np.ndarray: Combined quantized color values.
  """
  # Extract color channels
  B = image[:,:,0]
  G = image[:,:,1]
  R = image[:,:,2]

  # Quantize each color channel
  B_quant = colorQuantization(B, B_bins)
  G_quant = colorQuantization(G, G_bins)
  R_quant = colorQuantization(R, R_bins)

  # Combine quantized values
  combine_quant = (B_bins * G_bins * R_quant) + (B_bins * G_quant) + B_quant
  return combine_quant

def edgeQuantization(image, binsTheta):
  """
  Perform edge detection using Sobel operators and quantize edge orientations.

  Parameters:
  - image (np.ndarray): Input image array.
  - bins_theta (int): Number of bins for edge orientation quantization.

  Returns:
  - np.ndarray: Quantized edge orientations.
  """
  # Extract color channels
  B = image[:,:,0]
  G = image[:,:,1]
  R = image[:,:,2]

  # Sobel operations
  Bx = cv2.Sobel(B,cv2.CV_64F,1,0,ksize=3)
  By = cv2.Sobel(B,cv2.CV_64F,0,1,ksize=3)
  Gx = cv2.Sobel(G,cv2.CV_64F,1,0,ksize=3)
  Gy = cv2.Sobel(G,cv2.CV_64F,0,1,ksize=3)
  Rx = cv2.Sobel(R,cv2.CV_64F,1,0,ksize=3)
  Ry = cv2.Sobel(R,cv2.CV_64F,0,1,ksize=3)

  # |a| and |b|
  a = np.sqrt(Bx**2 + Gx**2 + Rx**2)
  b = np.sqrt(By**2 + Gy**2 + Ry**2)

  # ab
  ab = (Bx*By) + (Gx*Gy) + (Rx*Ry)

  # Image orientation
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

def colorTexton(h, w, colorQuant, color_img):
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
        
  return color_img

def edgeTexton(h, w, edgeQuant, edge_img):
  for i in range(0, h, 2):
    for j in range(0, w, 2):
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
  
  return edge_img

def textonSearch(colorQuant, cBins, edgeQuant, eBins):
  """
  Perform texture analysis using color and edge quantization, texton search, and histogram creation.

  Parameters:
  - color_quant (np.ndarray): Quantized color image.
  - c_bins (int): Number of bins for color histogram.
  - edge_quant (np.ndarray): Quantized edge image.
  - e_bins (int): Number of bins for edge histogram.

  Returns:
  - List[float]: Texton from combined color and edge features.
  """
  # Adjust image dimensions for odd sizes
  (h, w) = colorQuant.shape
  if (h % 2 == 1):
    colorQuant = colorQuant[:-1, :]
    edgeQuant = edgeQuant[:-1, :]

  if (w % 2 == 1):
    colorQuant = colorQuant[:, :-1]
    edgeQuant = edgeQuant[:, :-1]

  # Variable preparation
  (h, w) = colorQuant.shape
  color_img = np.zeros((h, w))
  edge_img = np.zeros((h, w))
  
  # texton search
  color_img = colorTexton(h, w, colorQuant, color_img)
  edge_img = edgeTexton(h, w, edgeQuant, edge_img)

  # Make color histogram
  cF = np.histogram(color_img.ravel(), cBins, [0, 64])
  colorFeatures = (np.array(cF[0]) / 6)  # perlu dibagi dg 6 meyesuaikan dg jumlah type texton yg digunakan

  # Make edge histogram
  eF = np.histogram(edge_img.ravel(), eBins, [0, 18])
  edgeFeatures = (np.array(eF[0]) / 6)

  # Combine color and edge features
  features = []
  features.extend(colorFeatures)
  features.extend(edgeFeatures)

  return features

def GLCM(image):
  """
  Perform Gray Level Co-occurrence Matrix using energy, contrast, correlation and entropy from co-occurrence matrix.
  
  Parameters:
  - image (np.ndarray): Input image array.
  
  Returns:
  - Tupel of np.ndarray.
  """
  # Convert image to greyscale
  grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Make co-occurance matrix
  gm = graycomatrix(grey_img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, normed=True)
  (h, w) = gm[:, :, 0, 0].shape

  glcm_features = []

  # Calculate energy, contrast, correlation, entropy using scikit library

  energy = graycoprops(gm, 'energy')
  contrast = graycoprops(gm, 'contrast')
  correlation = graycoprops(gm, 'correlation')
  glcm_features.extend(energy.tolist())
  glcm_features.extend(contrast.tolist())
  glcm_features.extend(correlation.tolist())

  entropy = []
  for i in range(0, 4):
    e = np.abs(gm[:, :, 0, i] * np.log2(gm[:, :, 0, i]))
    e[np.isnan(e)] = 0
    entropy.append(np.sum(e))

  return glcm_features, entropy

def mtcd(image):
  """
  Perform texture analysis using color and edge quantization, texton search, and GLCM.

  Parameters:
  - image (np.ndarray): Input image array.

  Returns:
  - np.ndarray: MTCD features.
  """
  # Define the bins for quantization
  colorBins = 64
  R_Bins = 4
  G_Bins = 4
  B_Bins = 4
  edgeBins = 18

  # Color Quantization
  colorQuant = combineColorQuantization(image, B_Bins, G_Bins, R_Bins)

  # Edge Quantization
  edgeQuant = edgeQuantization(image, edgeBins)

  # Texton Search
  features = textonSearch(colorQuant, colorBins, edgeQuant, edgeBins)

  # GLCM
  glcm, en = GLCM(image)

  # Make MTCD feature dimention to 1x98
  features.extend(glcm[0])
  features.extend(glcm[1])
  features.extend(glcm[2])
  features.extend(en)

  return np.array(features,dtype=object)