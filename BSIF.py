import scipy
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

from numba import jit, cuda


# @jit(target_backend='cuda')  
# @jit(target_backend='cuda', nopython=True)  
def zeroAround(img, c):
  padded_img = cv2.copyMakeBorder(img, c, c, c, c, cv2.BORDER_CONSTANT, None, 0)
  return padded_img

# @jit(target_backend='cuda')  
def bsif(img, filter_path, kname_split):
  filter_shape = kname_split[1]
  filter_size = filter_shape.split("x")[0]
  filter_size = int(filter_size)

  bit = int(kname_split[-1].replace('bit.mat', ''))
  # target = "ICAtextureFilters_{0}x{1}_{2}bit.mat".format(filter_size, filter_size, bit)
  # filter_path = os.path.join(filter_path,target)
  # print(filter_path)
  filter = scipy.io.loadmat(filter_path)['ICAtextureFilters']

  #padding
  image_size = img.shape[0]
  pad = int((filter_size-1)/2)
  r=pad

  # upimg = img[0:r,:]
  # btimg = img[-r:,:]
  # lfimg = img[:,0:r]
  # rtimg = img[:,-r:]

  # cr11 = img[0:r,0:r]
  # cr12 = img[0:r,-r:]
  # cr21 = img[-r:,0:r]
  # cr22 = img[-r:,-r:]

  # imgWrap = np.vstack((np.hstack((cr22, btimg, cr21)), np.hstack((rtimg, img, lfimg)), np.hstack((cr12, upimg, cr11))))
  imgWrap = np.vstack((np.hstack((img[-r:,-r:], img[-r:,:], img[-r:,0:r])), np.hstack((img[:,-r:], img, img[:,0:r])), np.hstack((img[0:r,-r:], img[0:r,:], img[0:r,0:r]))))
  # print('+++++++++++')
  # cv2_imshow(outwrap)
  # print('------')
  # cv2_imshow(imgWrap)
  # print('+++++++++++')

  # features = np.zeros((num_filter,1))
  # features = np.zeros((num_filter,10))
  features = []

  # cv2_imshow(padded_img)
  # cv2_imshow(imgWrap)

  binary_img = np.zeros((image_size, image_size, bit))
  # print(padded_img.shape)
  # print('!!!!!!!!!!!!')
  # print(binary_img.shape)
  abc = bit+1
  for i in range(1,abc):
    c2 = scipy.signal.convolve2d(imgWrap, np.rot90(filter[:,:,bit-i],2), mode='valid')
    binary_img[:,:, i-1] = c2>0

  for i in range(1,abc):
    padding_img = zeroAround(binary_img[:,:, i-1], pad)
    convBinary = scipy.signal.convolve2d(padding_img, np.rot90(filter[:,:,bit-i],2), mode='valid')
    counts, bins, bars = plt.hist(convBinary.ravel())
    # print(counts.shape)
    # features.append(counts)
    # features.resize((num_filter, len(counts)))
    for j in counts:
      features.append(j)
    # print(counts)
    # print('000000000')
    # print(counts.shape)
  # print('features.shape')
  # print(features.shape)
  # print('=====')
  # flatten_features = features.flatten()
  # flatten_features = [x or 0 for x in features.flatten()]
  # print(flatten_features)
  # print(flatten_features.shape)
  # print(features)
  # return flatten_features
  features = np.array(features)
  return features

# @jit(target_backend='cuda', nopython=False)  
def bsif2(img, filter_path, kname_split):
  filter_shape = kname_split[1]
  filter_size = int(filter_shape.split("x")[0])
  bit = int(kname_split[-1].replace('bit.mat', ''))
  filter = scipy.io.loadmat(filter_path)['ICAtextureFilters']

  #padding
  image_size = img.shape[0]
  r = int((filter_size-1)/2)

  imgWrap = np.vstack((np.hstack((img[-r:,-r:], img[-r:,:], img[-r:,0:r])), np.hstack((img[:,-r:], img, img[:,0:r])), np.hstack((img[0:r,-r:], img[0:r,:], img[0:r,0:r]))))
  
  # del img

  features = []

  binary_img = np.zeros((image_size, image_size, bit))

  for i in range(1,bit+1):
    c2 = scipy.signal.convolve2d(imgWrap, np.rot90(filter[:,:,bit-i],2), mode='valid')
    binary_img[:,:, i-1] = c2>0

  # del imgWrap

  for i in range(1,bit+1):
    padding_img = zeroAround(binary_img[:,:, i-1], r)
    convBinary = scipy.signal.convolve2d(padding_img, np.rot90(filter[:,:,bit-i],2), mode='valid')
    counts, bins, bars = plt.hist(convBinary.ravel())
    for j in counts:
      features.append(j)

  # del binary_img

  features = np.array(features)
  return features