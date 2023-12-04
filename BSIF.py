import scipy
import numpy as np
import cv2
from matplotlib import pyplot as plt

import cupy as cp
from cupyx.scipy.signal import convolve2d

def zeroAround(img, c):
  # Pad the input image with zero values around its borders
  padded_img = cv2.copyMakeBorder(img, c, c, c, c, cv2.BORDER_CONSTANT, None, 0)
  return padded_img

def bsif(img, filter_path, kname_split):
  filter_shape = kname_split[1]
  filter_size = int(filter_shape.split("x")[0])
  bit = int(kname_split[-1].replace('bit.mat', ''))
  filter = scipy.io.loadmat(filter_path)['ICAtextureFilters']

  #padding
  image_size = img.shape[0]
  r = int((filter_size-1)/2)

  imgWrap = np.vstack((np.hstack((img[-r:,-r:], img[-r:,:], img[-r:,0:r])), np.hstack((img[:,-r:], img, img[:,0:r])), np.hstack((img[0:r,-r:], img[0:r,:], img[0:r,0:r]))))

  features = []

  binary_img = np.zeros((image_size, image_size, bit))

  for i in range(1,bit+1):
    c2 = scipy.signal.convolve2d(imgWrap, np.rot90(filter[:,:,bit-i],2), mode='valid')
    binary_img[:,:, i-1] = c2>0

  for i in range(1,bit+1):
    padding_img = zeroAround(binary_img[:,:, i-1], r)
    convBinary = scipy.signal.convolve2d(padding_img, np.rot90(filter[:,:,bit-i],2), mode='valid')
    counts, bins, bars = plt.hist(convBinary.ravel())
    for j in counts:
      features.append(j)

  features = np.array(features)
  return features



def bsif_gpu(img, filter_path, kname_split):
  filter_shape = kname_split[1]
  filter_size = int(filter_shape.split("x")[0])
  bit = int(kname_split[-1].replace('bit.mat', ''))
  filter = scipy.io.loadmat(filter_path)['ICAtextureFilters']
  
  image_size = img.shape[0]
  r = int((filter_size-1)/2)

  imgWrap = cp.array(np.vstack((np.hstack((img[-r:,-r:], img[-r:,:], img[-r:,0:r])), np.hstack((img[:,-r:], img, img[:,0:r])), np.hstack((img[0:r,-r:], img[0:r,:], img[0:r,0:r])))))

  features = []

  binary_img = cp.array(np.zeros((image_size, image_size, bit)))

  for i in range(1,bit+1):
    c2 = convolve2d(imgWrap, cp.array(np.rot90(filter[:,:,bit-i],2)), mode='valid')
    binary_img[:,:, i-1] = c2>0
  
  for i in range(1,bit+1):
    padding_img = cp.array(zeroAround(cp.asnumpy(binary_img[:,:, i-1]), r))
    convBinary = convolve2d(padding_img, cp.array(np.rot90(filter[:,:,bit-i],2)), mode='valid')
    counts, _, _ = plt.hist(convBinary.get().ravel())
    for j in counts:
      features.append(j)
  
  features = np.array(features)
  return features