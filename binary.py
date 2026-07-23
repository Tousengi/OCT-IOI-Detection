import os
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd

input_folder = "input/dir"
output_folder = "output/dir"
save_binary = True
threshold_bin = 192
sigma = 20

def binarize_image(img_path, save_path=None, threshold=192):
  img=Image.open(img_path).convert('L')
  bin_img = img.point(lambda x: 255 if x < threhold else 0, 'L')

  if save_path:
    bin_img.save(save_path)
  return np.array(bin_img)

def curvature_analysis_from_array(arr, sigma=3, threshold=128):
  h, w = arr.shape
  upper_boundary = np.full(w, np.nan)

  for x in range(w):
    col = arr[:, x]
    indices = np.where(col > threshold)[0]
    if len(indices) > 0:
      upper_boundary[x] = indices[0]

  y_series = pd.Series(upper_boundary)
  y_filled = y_series.interpolate(method='linear', limit_direction='both').to_numpy()

  if np.all(np.isnan(y_filled)):
    return np.nan, np.nan

  y_smooth = gaussian_filter1d(y_filled, sigma=sigma)

  dy = np.gradient(y_smooth)
  d2y = np.gradient(dy)
  
  curvature = np.abs(d2y) / np.power(1 + np.power(dy, 2), 1.5)
  
  ds = np.sqrt(1 + np.power(dy, 2))
  
  total_curvature = np.sum(curvature * ds)
  total_length = np.sum(ds)
  mean_curvature = total_curvature / total_length if total_length > 0 else 0
  
  return mean_curvature

results1 = []

for filename in os.listdir(input_folder):
  img_path = os.path.join(input_folder, filename)
  save_path = os.path.join(output_folder, filename) if save_binary else None

  bin_arr = binarize_image(img_path, save_path, threshold=threshold_bin)
