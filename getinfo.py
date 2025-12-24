# getinfo.py
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11 Model Info Script (Params/FLOPs)')
    # 通用参数（与 train.py 对齐/兼容）
    parser.add_argument('--cfg', type=str, default='/home/user/PROJECT/FSWD/FSW-MERGE/data.yaml',
                        help='数据集配置文件路径（本脚本不使用，仅保留兼容）')
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/11/yolo11s.yaml',
                        help='模型配置/权重路径（.yaml / .pt）')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size（用于 FLOPs 估算，默认 1）')
    parser.add_argument('--img-size', type=int, default=640, help='输入图片大小（imgsz）')
    parser.add_argument('--device', type=str, default='0', help='CUDA 设备，如 0 或 cpu')

    # 输出控制
    parser.add_argument('--project', type=str, default='runs/info', help='结果保存目录')
    parser.add_argument('--name', type=str, default='model_info', help='实验名')
    parser.add_argument('--save', action='store_true', help='保存信息到文本文件')
    parser.add_argument('--verbose', action='store_true', help='打印更详细的结构信息（若可用）')
    return parser.parse_args()


def _select_device(device_str: str):
    d = str(device_str).strip().lower()
    if d == 'cpu' or d == '':
        return torch.device('cpu')
    # ultralytics 常用的是 "0"/"0,1"；此处仅选第一个用于 info
    first = d.split(',')[0].strip()
    if torch.cuda.is_available():
        try:
            idx = int(first)
            return torch.device(f'cuda:{idx}')
        except Exception:
            return torch.device('cuda:0')
    return torch.device('cpu')


def _format_num(n: float) -> str:
    # 统一格式化：参数量等用 K/M/G
    n = float(n)
    for unit in ['','K','M','G','T']:
        if abs(n) < 1000.0:
            return f'{n:.3f}{unit}'
        n /= 1000.0
    return f'{n:.3f}P'


def _model_param_stats(torch_model: torch.nn.Module):
    total = sum(p.numel() for p in torch_model.parameters())
    trainable = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in torch_model.buffers())
    return total, trainable, buffers


def _estimate_flops_with_thop(torch_model: torch.nn.Module, imgsz: int, batch: int, device: torch.device):
    # 使用 thop 估算 FLOPs/MACs（可选依赖）
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None, None, '未安装 thop，跳过 FLOPs 估算（可选：pip install thop）'

    try:
        torch_model.eval()
        x = torch.zeros((batch, 3, imgsz, imgsz), device=device)
        # thop 返回的是 MACs（多数情况下），这里同时原样输出
        macs, params = profile(torch_model, inputs=(x,), verbose=False)
        return macs, params, None
    except Exception as e:
        return None, None, f'thop 估算失败：{e}'


def main():
    args = parse_args()
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(args.device)

    # 加载 YOLO（yaml 或 pt）
    yolo = YOLO(args.model)

    # 获取底层 torch.nn.Module（不同版本 ultralytics 可能字段不同）
    torch_model = getattr(yolo, 'model', None)
    if torch_model is None:
        raise RuntimeError('无法从 YOLO 实例获取 torch 模型（yolo.model 为空）')

    torch_model.to(device)

    total_params, trainable_params, buffer_elems = _model_param_stats(torch_model)

    # 模型大小（仅参数，不含优化器等）：numel * dtype_bytes
    # 这里以当前参数 dtype 计算一个近似值（若混合 dtype，则按逐个参数累加）
    bytes_params = 0
    for p in torch_model.parameters():
        bytes_params += p.numel() * torch.tensor([], dtype=p.dtype).element_size()
    size_mb = bytes_params / (1024 ** 2)

    # FLOPs/MACs（可选）
    macs, thop_params, flops_note = _estimate_flops_with_thop(
        torch_model=torch_model,
        imgsz=int(args.img_size),
        batch=int(max(1, args.batch_size)),
        device=device,
    )

    lines = []
    lines.append(f'Model: {args.model}')
    lines.append(f'Device: {device}')
    lines.append(f'Input: batch={int(max(1, args.batch_size))}, 3x{args.img_size}x{args.img_size}')
    lines.append('')
    lines.append(f'Total params: {total_params} ({_format_num(total_params)})')
    lines.append(f'Trainable params: {trainable_params} ({_format_num(trainable_params)})')
    lines.append(f'Buffers (elements): {buffer_elems} ({_format_num(buffer_elems)})')
    lines.append(f'Param size (approx): {size_mb:.2f} MiB')

    if macs is not None:
        # thop 常见返回 MACs；部分人把 FLOPs 约等于 2*MACs（依实现而异）
        lines.append('')
        lines.append(f'MACs (thop): {macs:.0f} ({_format_num(macs)})')
        lines.append(f'FLOPs (approx=2*MACs): {2.0 * macs:.0f} ({_format_num(2.0 * macs)})')
    else:
        lines.append('')
        lines.append(f'FLOPs/MACs: N/A（{flops_note}）')

    if args.verbose:
        # ultralytics 常见有 info()；若不可用则忽略
        try:
            lines.append('')
            lines.append('--- ultralytics model.info() ---')
            # 部分版本直接打印；返回值不固定，这里尽量兼容
            info_ret = yolo.info(verbose=True)
            if isinstance(info_ret, str) and info_ret.strip():
                lines.append(info_ret.strip())
        except Exception as e:
            lines.append('')
            lines.append(f'info(verbose=True) 不可用：{e}')

    text = '\n'.join(lines)
    print(text)

    if args.save:
        out_path = save_dir / 'model_info.txt'
        out_path.write_text(text, encoding='utf-8')
        print(f'\nSaved to: {out_path}')


if __name__ == '__main__':
    main()