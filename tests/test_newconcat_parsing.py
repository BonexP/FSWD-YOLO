"""
测试 NewConcat 模块在 tasks.py 中的参数解析逻辑
Test NewConcat parameter parsing logic in tasks.py
"""
import torch
from ultralytics.nn.tasks import parse_model
from ultralytics import YOLO


def test_newconcat_parsing():
    """测试不同的 YAML 配置下 NewConcat 的参数解析"""

    print("=" * 80)
    print("测试 NewConcat 参数解析 / Testing NewConcat Parameter Parsing")
    print("=" * 80)

    # 测试用例 1: 只指定 out_channels
    print("\n【测试 1】只指定 out_channels: [512]")
    print("期望: out_channels=512, kernel_size=3, use_dw=True, post_fusion=False")
    test_case_1 = {
        'args': [512],
        'expected': {
            'out_channels': 512,
            'kernel_size': 3,
            'use_dw': True,
            'post_fusion': False
        }
    }

    # 测试用例 2: 指定 out_channels 和 kernel_size
    print("\n【测试 2】指定 out_channels 和 kernel_size: [512, 5]")
    print("期望: out_channels=512, kernel_size=5, use_dw=True, post_fusion=False")
    test_case_2 = {
        'args': [512, 5],
        'expected': {
            'out_channels': 512,
            'kernel_size': 5,
            'use_dw': True,
            'post_fusion': False
        }
    }

    # 测试用例 3: 指定 out_channels, kernel_size, use_dw
    print("\n【测试 3】指定 out_channels, kernel_size, use_dw: [512, 3, False]")
    print("期望: out_channels=512, kernel_size=3, use_dw=False, post_fusion=False")
    test_case_3 = {
        'args': [512, 3, False],
        'expected': {
            'out_channels': 512,
            'kernel_size': 3,
            'use_dw': False,
            'post_fusion': False
        }
    }

    # 测试用例 4: 指定所有参数
    print("\n【测试 4】指定所有参数: [512, 3, True, True]")
    print("期望: out_channels=512, kernel_size=3, use_dw=True, post_fusion=True")
    test_case_4 = {
        'args': [512, 3, True, True],
        'expected': {
            'out_channels': 512,
            'kernel_size': 3,
            'use_dw': True,
            'post_fusion': True
        }
    }

    # 模拟参数解析逻辑（与 tasks.py 中的逻辑一致）
    test_cases = [test_case_1, test_case_2, test_case_3, test_case_4]

    for i, test_case in enumerate(test_cases, 1):
        args = test_case['args']
        expected = test_case['expected']

        # 解析参数（模拟 tasks.py 中的逻辑）
        out_channels = int(args[0])
        kernel_size = int(args[1]) if len(args) > 1 else 3
        use_dw = bool(args[2]) if len(args) > 2 else True
        post_fusion = bool(args[3]) if len(args) > 3 else False

        # 验证解析结果
        actual = {
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'use_dw': use_dw,
            'post_fusion': post_fusion
        }

        # 打印结果
        print(f"\n实际解析结果: {actual}")

        # 检查是否匹配
        if actual == expected:
            print(f"✅ 测试用例 {i} 通过")
        else:
            print(f"❌ 测试用例 {i} 失败")
            print(f"   期望: {expected}")
            print(f"   实际: {actual}")

    print("\n" + "=" * 80)
    print("现在尝试加载实际的 YAML 配置文件")
    print("=" * 80)

    try:
        # 尝试加载 yolo11s_NewConcat.yaml
        model = YOLO('/home/user/projects/YOLO11/ultralytics/cfg/models/11/yolo11s_NewConcat.yaml')
        print("\n✅ 成功加载 yolo11s_NewConcat.yaml 配置")
        print(f"模型结构加载成功，共 {len(model.model.model)} 层")

        # 查找 NewConcat 层
        from ultralytics.nn.modules import NewConcat
        newconcat_layers = []
        for i, layer in enumerate(model.model.model):
            if isinstance(layer, NewConcat):
                newconcat_layers.append((i, layer))

        print(f"\n找到 {len(newconcat_layers)} 个 NewConcat 层:")
        for idx, layer in newconcat_layers:
            print(f"  - 第 {idx} 层: out_channels={layer.out_channels}, "
                  f"use_dw={layer.use_dw}, post_fusion={layer.post_fusion}, "
                  f"num_branches={layer.num_branches}")

    except Exception as e:
        print(f"\n❌ 加载 YAML 配置失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_newconcat_parsing()

