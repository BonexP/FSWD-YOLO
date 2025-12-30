import torch
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11s_NewConcat.yaml").model.eval()

model.info(detailed=True)
x = torch.randn(1, 3, 640, 640, device="cpu")

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with torch.no_grad():
        _ = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

