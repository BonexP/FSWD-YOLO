# getinfo.py
import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLO Model Summary (ultralytics API)")
    p.add_argument("--model", type=str, default="ultralytics/cfg/models/11/yolo11s.yaml",
                   help="模型配置/权重路径（.yaml / .pt）")
    p.add_argument("--img-size", type=int, default=640, help="输入图片大小（imgsz）")
    p.add_argument("--device", type=str, default="0", help="CUDA 设备，如 0 或 cpu")
    p.add_argument("--verbose", action="store_true", help="更详细的结构信息（若版本支持）")
    return p.parse_args()


def main():
    args = parse_args()
    yolo = YOLO(args.model)

    # 1) 优先用 ultralytics 自带 summary/info
    # 不同版本可能是 yolo.info(...) 或 yolo.model.info(...)
    info_called = False

    for fn in (
        lambda: yolo.info(verbose=args.verbose),
        lambda: yolo.model.info(verbose=args.verbose),
        lambda: yolo.model.info(),  # 某些版本没有 verbose 参数
    ):
        try:
            ret = fn()
            info_called = True
            if isinstance(ret, str) and ret.strip():
                print(ret)
            break
        except Exception:
            pass

    if not info_called:
        print("当前 ultralytics 版本未暴露 `info()`/`model.info()`；改用手动统计参数量。")

    # 2) 手动补充：参数量（总/可训练）
    tm = getattr(yolo, "model", None)
    if tm is None:
        raise RuntimeError("无法从 YOLO 实例获取 torch 模型（yolo.model 为空）")

    total = sum(p.numel() for p in tm.parameters())
    trainable = sum(p.numel() for p in tm.parameters() if p.requires_grad)

    print(f"\nTotal params: {total}")
    print(f"Trainable params: {trainable}")

    # 备注：img-size/device 在 info() 里通常不参与 FLOPs 计算；如需 FLOPs 仍建议 thop/profile。
    _ = args.img_size, args.device


if __name__ == "__main__":
    main()