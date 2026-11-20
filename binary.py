import os
import re
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter1d, label as cc_label
from scipy.stats import spearmanr
import pandas as pd

input_folder = "input/dir"
output_folder = "output/dir"
development_folder = "dataset"
external_folder = "dataset/test_gun"
save_binary = True
threshold_bin = 110
min_area = 40
vitreous_top = 0.05
vitreous_bottom = 0.55
sigma = 20
n_folds = 10
cell_class = 0
extensions = (".png", ".jpg", ".jpeg")
thresholds = [100, 110, 120, 130, 135, 140, 150, 160, 170, 180, 190, 200, 210, 220]
areas = [3, 5, 10, 20, 40, 80, 120, 160, 200, 250]
reported = [(135, 3), (None, 3), (threshold_bin, min_area)]
name_pattern = re.compile(r"^([IN])-(\d+)([HV])(B?)\.(jpe?g|png)$", re.I)

def binarize_image(img_path, save_path=None, threshold=threshold_bin):
  img = Image.open(img_path).convert('L')
  bin_img = img.point(lambda x: 255 if x > threshold else 0, 'L')

  if save_path:
    bin_img.save(save_path)
  return np.array(bin_img)

def otsu(roi):
  hist = np.bincount(roi.ravel(), minlength=256).astype(float)
  w = np.cumsum(hist)
  m = np.cumsum(hist * np.arange(256))
  tw, tm = w[-1], m[-1]
  valid = (w > 0) & (w < tw)
  var = np.full(256, -1.0)
  var[valid] = (tm * w[valid] / tw - m[valid]) ** 2 / (w[valid] * (tw - w[valid]))
  return int(np.argmax(var))

def read_grayscale(img_path):
  return np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8)

def vitreous_roi(arr):
  h = arr.shape[0]
  return arr[int(h * vitreous_top):int(h * vitreous_bottom), :]

def object_count_from_array(arr, threshold=threshold_bin, area=min_area):
  labels, n = cc_label(vitreous_roi(arr) > threshold)
  if n == 0:
    return 0
  sizes = np.bincount(labels.ravel())[1:]
  return int((sizes >= area).sum())

def curvature_analysis_from_array(arr, sigma=sigma, threshold=threshold_bin):
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
    return 0.0

  y_smooth = gaussian_filter1d(y_filled, sigma=sigma)

  dy = np.gradient(y_smooth)
  d2y = np.gradient(dy)

  curvature = np.abs(d2y) / np.power(1 + np.power(dy, 2), 1.5)

  ds = np.sqrt(1 + np.power(dy, 2))

  total_curvature = np.sum(curvature * ds)
  total_length = np.sum(ds)
  mean_curvature = total_curvature / total_length if total_length > 0 else 0

  return mean_curvature

def features(img_path, threshold=threshold_bin, area=min_area):
  arr = read_grayscale(img_path)
  if threshold is None:
    threshold = otsu(vitreous_roi(arr))
  return (object_count_from_array(arr, threshold, area),
          curvature_analysis_from_array(arr, sigma, threshold))

def development_pairs(fold_root, folds=n_folds):
  pairs = []
  for i in range(1, folds + 1):
    image_dir = os.path.join(fold_root, "fold%d" % i, "val", "images")
    label_dir = os.path.join(fold_root, "fold%d" % i, "val", "labels")
    if not os.path.isdir(image_dir):
      continue
    for name in sorted(os.listdir(image_dir)):
      if not name.lower().endswith(extensions):
        continue
      label_path = os.path.join(label_dir, os.path.splitext(name)[0] + ".txt")
      n_cells = 0
      if os.path.exists(label_path):
        with open(label_path) as handle:
          for line in handle:
            if line.strip() and int(float(line.split()[0])) == cell_class:
              n_cells += 1
      pairs.append((os.path.join(image_dir, name), n_cells))
  return pairs

def grid_search(pairs):
  annotated = np.array([n for _, n in pairs], dtype=float)
  detections = {(t, a): [] for t in thresholds for a in areas}
  for path, _ in pairs:
    roi = vitreous_roi(read_grayscale(path))
    for t in thresholds:
      labels, n = cc_label(roi > t)
      sizes = np.bincount(labels.ravel())[1:] if n else np.zeros(0, dtype=np.int64)
      for a in areas:
        detections[(t, a)].append(int((sizes >= a).sum()))
  records = []
  for (t, a), counts in detections.items():
    counts = np.array(counts, dtype=float)
    records.append(dict(threshold=t,
                        min_area=a,
                        spearman=float(spearmanr(counts, annotated).correlation),
                        detections_per_bscan=float(counts.mean()),
                        annotated_per_bscan=float(annotated.mean())))
  return pd.DataFrame(records).sort_values("spearman", ascending=False).reset_index(drop=True)

def visit_scores(paths, threshold=threshold_bin, area=min_area):
  values = [features(p, threshold, area) for p in paths]
  return (float(np.mean([v[0] for v in values])),
          float(np.mean([v[1] for v in values])))

def external_comparison(image_folder, threshold=threshold_bin, area=min_area):
  visits = {}
  for name in sorted(os.listdir(image_folder)):
    match = name_pattern.match(name)
    if not match:
      continue
    group, number, _, is_baseline, _ = match.groups()
    key = (group.upper(), int(number))
    visits.setdefault(key, {"baseline": [], "post": []})
    visits[key]["baseline" if is_baseline else "post"].append(os.path.join(image_folder, name))
  rows = []
  for key in sorted(visits, key=lambda k: (k[0] != "I", k[1])):
    baseline = visit_scores(visits[key]["baseline"], threshold, area)
    post = visit_scores(visits[key]["post"], threshold, area)
    rows.append(dict(case=key[1] if key[0] == "I" else 20 + key[1],
                     ioi=1 if key[0] == "I" else 0,
                     delta_objects=post[0] - baseline[0],
                     delta_curvature=post[1] - baseline[1]))
  table = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
  table["prediction"] = ((table.delta_objects > 0) | (table.delta_curvature > 0)).astype(int)
  return table

results1 = []

if os.path.isdir(input_folder):
  for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(extensions):
      continue
    img_path = os.path.join(input_folder, filename)
    save_path = os.path.join(output_folder, filename) if save_binary else None

    if save_path:
      os.makedirs(output_folder, exist_ok=True)
      binarize_image(img_path, save_path, threshold=threshold_bin)

    n_objects, mean_curvature = features(img_path, threshold_bin, min_area)
    results1.append(dict(image=filename, objects=n_objects, curvature=mean_curvature))

  pd.DataFrame(results1).to_csv("baseline_features.csv", index=False)

pairs = development_pairs(development_folder)
if pairs:
  table = grid_search(pairs)
  table.to_csv("baseline_grid_search.csv", index=False)
  best = table.iloc[0]
  print("development B-scans %d, annotated cells %d, annotated per B-scan %.2f"
        % (len(pairs), int(sum(n for _, n in pairs)), best.annotated_per_bscan))
  print("best threshold %d, minimum area %d, spearman %.3f, detections per B-scan %.2f"
        % (best.threshold, best.min_area, best.spearman, best.detections_per_bscan))

if pairs:
  annotated = np.array([n for _, n in pairs], dtype=float)
  otsu_counts = np.array([features(p, None, 3)[0] for p, _ in pairs], dtype=float)
  print("otsu threshold, minimum area 3: spearman %.3f, detections per B-scan %.2f"
        % (spearmanr(otsu_counts, annotated).correlation, otsu_counts.mean()))

if os.path.isdir(external_folder):
  for threshold, area in reported:
    tag = "otsu" if threshold is None else "threshold%d" % threshold
    external = external_comparison(external_folder, threshold, area)
    external.to_csv("baseline_external_%s_area%d.csv" % (tag, area), index=False)
    tp = int(((external.ioi == 1) & (external.prediction == 1)).sum())
    fn = int(((external.ioi == 1) & (external.prediction == 0)).sum())
    tn = int(((external.ioi == 0) & (external.prediction == 0)).sum())
    fp = int(((external.ioi == 0) & (external.prediction == 1)).sum())
    print("%s, minimum area %d: sensitivity %d/%d, specificity %d/%d, accuracy %d/%d"
          % (tag, area, tp, tp + fn, tn, tn + fp, tp + tn, len(external)))
