from ultralytics import YOLO
import torch

SEED = 42
N_FOLDS = 10

for i in range(N_FOLDS):
    model = YOLO("configs/yolo11_proposed.yaml")
    model.train(
        data=f"dataset{i}.yaml",
        epochs=500,
        imgsz=1280,
        batch=24,
        patience=0,
        amp=False,
        device=[0, 1, 2],
        seed=SEED,
        deterministic=True,
        project="runs",
        name=f"fold{i}",
        exist_ok=True,
    )
    del model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
