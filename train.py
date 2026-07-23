from ultralytics import YOLO
import torch

for i in range(10):
    model = YOLO("train.yaml")
    train_results = model.train(
        data=f"dataset{i}.yaml",
        epochs=500,
        imgsz=1280,
        batch=32,
        patience=0,
        amp=False,
        device=[0, 1, 2, 3],
    )
    del model
    torch.cuda.empty_cache()
    torch.cuda_ipc_collect()
