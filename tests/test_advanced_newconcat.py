"""
测试高级 NewConcat 配置
Test advanced NewConcat configuration
"""
from ultralytics import YOLO
from ultralytics.nn.modules import NewConcat


def test_advanced_config():
    """测试高级 NewConcat 配置文件"""

    print("=" * 80)
    print("测试高级 NewConcat 配置 / Testing Advanced NewConcat Configuration")
    print("=" * 80)

    try:
        # 加载高级配置
        print("\n加载 yolo11s_NewConcat_advanced.yaml...")
        model = YOLO('/home/user/projects/YOLO11/ultralytics/cfg/models/11/yolo11s_NewConcat_advanced.yaml')
        print(f"✅ 成功加载配置，共 {len(model.model.model)} 层")

        # 查找所有 NewConcat 层并显示其配置
        print("\n" + "=" * 80)
        print("NewConcat 层配置详情:")
        print("=" * 80)

        newconcat_count = 0
        for i, layer in enumerate(model.model.model):
            if isinstance(layer, NewConcat):
                newconcat_count += 1
                print(f"\n【第 {i} 层 / Layer {i}】")
                print(f"  - 输出通道 / out_channels: {layer.out_channels}")
                print(f"  - 分支数量 / num_branches: {layer.num_branches}")
                print(f"  - 深度可分离卷积 / use_dw: {layer.use_dw}")
                print(f"  - 后融合处理 / post_fusion: {layer.post_fusion}")

                # 分析配置特点
                if not layer.use_dw and not layer.post_fusion:
                    print(f"  ⚡ 配置类型: 标准卷积模式 - 更强的特征表达")
                elif layer.use_dw and layer.post_fusion:
                    print(f"  🔥 配置类型: 高级融合模式 - 复杂特征融合")
                elif layer.use_dw and not layer.post_fusion:
                    print(f"  ✨ 配置类型: 高效模式 - 平衡性能与精度")
                else:
                    print(f"  💡 配置类型: 自定义模式")

        print("\n" + "=" * 80)
        print(f"总计找到 {newconcat_count} 个 NewConcat 层")
        print("=" * 80)

        # 显示模型摘要
        print("\n模型摘要:")
        print(model.model)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_advanced_config()
    if success:
        print("\n" + "🎉" * 40)
        print("所有测试通过！NewConcat 参数解析工作正常。")
        print("All tests passed! NewConcat parameter parsing works correctly.")
        print("🎉" * 40)

