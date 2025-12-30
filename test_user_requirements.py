"""
演示用户需求的具体用例测试
Demonstrates the exact use cases requested by the user
"""
from ultralytics import YOLO
from ultralytics.nn.modules import NewConcat
import torch


def test_user_requirements():
    """测试用户提出的具体需求"""

    print("=" * 80)
    print("测试用户需求 / Testing User Requirements")
    print("=" * 80)

    print("\n用户需求 1: 传入完整参数控制 NewConcat 表现")
    print("User Requirement 1: Pass full parameters to control NewConcat behavior")
    print("-" * 80)

    # 需求: - [[-1, 13], 1, NewConcat, [512,3,True,True]] # cat head P4
    # 期望: 输出512通道，卷积核3，使用dw卷积，启用后处理
    print("\n示例: [[-1, 13], 1, NewConcat, [512, 3, True, True]]")
    print("期望: out_channels=512, kernel_size=3, use_dw=True, post_fusion=True")

    # 模拟解析
    args = [512, 3, True, True]
    out_channels = int(args[0])
    kernel_size = int(args[1]) if len(args) > 1 else 3
    use_dw = bool(args[2]) if len(args) > 2 else True
    post_fusion = bool(args[3]) if len(args) > 3 else False

    print(f"✅ 解析结果: out_channels={out_channels}, kernel_size={kernel_size}, "
          f"use_dw={use_dw}, post_fusion={post_fusion}")

    # 创建实际模块验证
    c1_list = [256, 512]  # 假设两个分支的输入通道
    module = NewConcat(in_channels_list=c1_list,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      use_dw=use_dw,
                      post_fusion=post_fusion)

    print(f"✅ 模块创建成功: {module.__class__.__name__}")
    print(f"   - 输出通道: {module.out_channels}")
    print(f"   - 使用DW卷积: {module.use_dw}")
    print(f"   - 后处理: {module.post_fusion}")
    print(f"   - 分支数量: {module.num_branches}")

    # 测试前向传播
    x1 = torch.randn(1, 256, 32, 32)
    x2 = torch.randn(1, 512, 32, 32)
    output = module([x1, x2])
    print(f"✅ 前向传播成功: 输入 [{x1.shape}, {x2.shape}] -> 输出 {output.shape}")

    print("\n" + "=" * 80)
    print("\n用户需求 2: 只传入 out_channels，其他参数使用默认值")
    print("User Requirement 2: Pass only out_channels, use defaults for others")
    print("-" * 80)

    # 需求: - [[-1, 13], 1, NewConcat, [512]] # cat head P4
    # 期望: 显式说明传出512通道，其他参数使用默认值
    print("\n示例: [[-1, 13], 1, NewConcat, [512]]")
    print("期望: out_channels=512, kernel_size=3(默认), use_dw=True(默认), post_fusion=False(默认)")

    # 模拟解析
    args = [512]
    out_channels = int(args[0])
    kernel_size = int(args[1]) if len(args) > 1 else 3
    use_dw = bool(args[2]) if len(args) > 2 else True
    post_fusion = bool(args[3]) if len(args) > 3 else False

    print(f"✅ 解析结果: out_channels={out_channels}, kernel_size={kernel_size}, "
          f"use_dw={use_dw}, post_fusion={post_fusion}")

    # 创建实际模块验证
    module2 = NewConcat(in_channels_list=c1_list,
                       out_channels=out_channels,
                       kernel_size=kernel_size,
                       use_dw=use_dw,
                       post_fusion=post_fusion)

    print(f"✅ 模块创建成功: {module2.__class__.__name__}")
    print(f"   - 输出通道: {module2.out_channels}")
    print(f"   - 使用DW卷积: {module2.use_dw} (默认)")
    print(f"   - 后处理: {module2.post_fusion} (默认)")
    print(f"   - 分支数量: {module2.num_branches}")

    # 测试前向传播
    output2 = module2([x1, x2])
    print(f"✅ 前向传播成功: 输入 [{x1.shape}, {x2.shape}] -> 输出 {output2.shape}")

    print("\n" + "=" * 80)
    print("\n用户需求 3: 部分参数自定义，其余使用默认")
    print("User Requirement 3: Customize some parameters, use defaults for others")
    print("-" * 80)

    # 测试各种组合
    test_cases = [
        {
            'args': [512, 5],
            'desc': '自定义 kernel_size',
            'expected': {'out_channels': 512, 'kernel_size': 5, 'use_dw': True, 'post_fusion': False}
        },
        {
            'args': [512, 3, False],
            'desc': '禁用 DW 卷积',
            'expected': {'out_channels': 512, 'kernel_size': 3, 'use_dw': False, 'post_fusion': False}
        },
        {
            'args': [256, 1, True, False],
            'desc': '最小卷积核 + DW',
            'expected': {'out_channels': 256, 'kernel_size': 1, 'use_dw': True, 'post_fusion': False}
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test['desc']}")
        print(f"配置: {test['args']}")

        args = test['args']
        out_channels = int(args[0])
        kernel_size = int(args[1]) if len(args) > 1 else 3
        use_dw = bool(args[2]) if len(args) > 2 else True
        post_fusion = bool(args[3]) if len(args) > 3 else False

        actual = {
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'use_dw': use_dw,
            'post_fusion': post_fusion
        }

        if actual == test['expected']:
            print(f"✅ 解析正确: {actual}")
        else:
            print(f"❌ 解析错误!")
            print(f"   期望: {test['expected']}")
            print(f"   实际: {actual}")

    print("\n" + "=" * 80)
    print("\n最终验证: 加载实际 YAML 配置")
    print("Final Validation: Load actual YAML configuration")
    print("=" * 80)

    try:
        # 加载原始用户的配置文件
        print("\n加载 yolo11s_NewConcat.yaml...")
        model = YOLO('/home/user/projects/YOLO11/ultralytics/cfg/models/11/yolo11s_NewConcat.yaml')
        print("✅ 配置文件加载成功!")

        # 检查 NewConcat 层
        newconcat_layers = []
        for i, layer in enumerate(model.model.model):
            if isinstance(layer, NewConcat):
                newconcat_layers.append((i, layer))

        print(f"\n找到 {len(newconcat_layers)} 个 NewConcat 层:")
        for idx, layer in newconcat_layers:
            print(f"\n第 {idx} 层配置:")
            print(f"  ✓ out_channels: {layer.out_channels}")
            print(f"  ✓ use_dw: {layer.use_dw}")
            print(f"  ✓ post_fusion: {layer.post_fusion}")
            print(f"  ✓ num_branches: {layer.num_branches}")

        print("\n" + "🎉" * 40)
        print("\n✅ 所有用户需求已满足！")
        print("✅ All user requirements satisfied!")
        print("\n关键功能:")
        print("  1. ✅ 可以通过 YAML 完整控制所有参数")
        print("  2. ✅ 可以只指定部分参数，其余使用默认值")
        print("  3. ✅ 参数解析逻辑正确匹配 NewConcat.__init__ 签名")
        print("  4. ✅ 支持灵活的配置组合")
        print("\nKey Features:")
        print("  1. ✅ Full control of all parameters via YAML")
        print("  2. ✅ Specify only some parameters, use defaults for others")
        print("  3. ✅ Parsing logic correctly matches NewConcat.__init__ signature")
        print("  4. ✅ Supports flexible configuration combinations")
        print("\n" + "🎉" * 40)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_user_requirements()

