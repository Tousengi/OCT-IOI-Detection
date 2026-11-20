from ultralytics import YOLO
import os
import re
import numpy as np
import pandas as pd
import ultralytics.nn.tasks as tasks
from cpm_module import CPMSigmoid

tasks.CPMSigmoid = CPMSigmoid

image_folder = "testset/images"
test_model_path = "test_model_path/last.pt"
conf_threshold = 0.25
iou_threshold = 0.7
image_size = 1280
cell_class = 0
irregularity_class = 1
n_positive = 20
name_pattern = re.compile(r"^([IN])-(\d+)([HV])(B?)\.(jpe?g|png)$", re.I)

model = YOLO(test_model_path)

counts = {}

for img in sorted(os.listdir(image_folder)):
  match = name_pattern.match(img)
  if not match:
    continue
  img_path = os.path.join(image_folder, img)
  results = model(img_path, conf=conf_threshold, iou=iou_threshold,
                  imgsz=image_size, verbose=False)
  classes = np.concatenate([r.boxes.cls.cpu().numpy() for r in results])

  group, number, _, is_baseline, _ = match.groups()
  key = (group.upper(), int(number), "baseline" if is_baseline else "post")
  scores = counts.setdefault(key, [0, 0])
  scores[0] += int((classes == cell_class).sum())
  scores[1] += int((classes == irregularity_class).sum())

rows = []

for group, number in sorted({(k[0], k[1]) for k in counts},
                            key=lambda k: (k[0] != "I", k[1])):
  baseline = counts.get((group, number, "baseline"), [0, 0])
  post = counts.get((group, number, "post"), [0, 0])
  rows.append(dict(case=number if group == "I" else n_positive + number,
                   ioi=1 if group == "I" else 0,
                   vcs_baseline=baseline[0],
                   vcs_post=post[0],
                   vriis_baseline=baseline[1],
                   vriis_post=post[1],
                   delta_vcs=post[0] - baseline[0],
                   delta_vriis=post[1] - baseline[1]))

table = pd.DataFrame(rows).sort_values("case").reset_index(drop=True)
table["prediction"] = ((table.delta_vcs > 0) | (table.delta_vriis > 0)).astype(int)
table.to_csv("external_scores.csv", index=False)

tp = int(((table.ioi == 1) & (table.prediction == 1)).sum())
fn = int(((table.ioi == 1) & (table.prediction == 0)).sum())
tn = int(((table.ioi == 0) & (table.prediction == 0)).sum())
fp = int(((table.ioi == 0) & (table.prediction == 1)).sum())

print("sensitivity %d/%d" % (tp, tp + fn))
print("specificity %d/%d" % (tn, tn + fp))
print("accuracy %d/%d" % (tp + tn, len(table)))
