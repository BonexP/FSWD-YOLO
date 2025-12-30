# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Test module for NewConcat implementation."""

import pytest
import torch

from ultralytics.nn.modules import NewConcat


class TestNewConcat:
    """Test cases for the NewConcat module."""

    def test_basic_functionality(self):
        """Test basic forward pass with same-sized inputs."""
        # Create module
        in_channels_list = [64, 128, 256]
        out_channels = 128
        module = NewConcat(in_channels_list, out_channels)
        
        # Create input tensors with same spatial size
        x_list = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 32, 32),
            torch.randn(2, 256, 32, 32),
        ]
        
        # Forward pass
        output = module(x_list)
        
        # Check output shape
        assert output.shape == (2, out_channels, 32, 32), f"Expected shape (2, {out_channels}, 32, 32), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_different_spatial_sizes(self):
        """Test with inputs of different spatial sizes."""
        in_channels_list = [64, 128]
        out_channels = 96
        module = NewConcat(in_channels_list, out_channels)
        
        # Create input tensors with different spatial sizes
        x_list = [
            torch.randn(2, 64, 16, 16),   # Smaller
            torch.randn(2, 128, 32, 32),  # Larger (will be target size)
        ]
        
        # Forward pass
        output = module(x_list)
        
        # Output should match the largest input size
        assert output.shape == (2, out_channels, 32, 32), f"Expected shape (2, {out_channels}, 32, 32), got {output.shape}"

    def test_upscaling(self):
        """Test spatial upscaling of smaller feature maps."""
        in_channels_list = [64, 128]
        out_channels = 80
        module = NewConcat(in_channels_list, out_channels)
        
        # Small and large feature maps
        x_list = [
            torch.randn(1, 64, 8, 8),     # Small (will be upscaled)
            torch.randn(1, 128, 16, 16),  # Large (target size)
        ]
        
        output = module(x_list)
        assert output.shape == (1, out_channels, 16, 16)

    def test_downscaling(self):
        """Test spatial downscaling of larger feature maps."""
        in_channels_list = [128, 64]
        out_channels = 96
        module = NewConcat(in_channels_list, out_channels)
        
        # Large and small feature maps
        x_list = [
            torch.randn(1, 128, 32, 32),  # Large (will be downscaled)
            torch.randn(1, 64, 16, 16),   # Small (target size - but large has more area)
        ]
        
        output = module(x_list)
        # Should align to the largest spatial size (32x32)
        assert output.shape == (1, out_channels, 32, 32)

    def test_depthwise_separable_conv(self):
        """Test with depthwise separable convolution enabled."""
        in_channels = 128
        out_channels = 128
        in_channels_list = [in_channels, in_channels]
        
        # With depthwise separable conv
        module_dw = NewConcat(in_channels_list, out_channels, use_dw=True)
        
        # Without depthwise separable conv
        module_std = NewConcat(in_channels_list, out_channels, use_dw=False)
        
        x_list = [
            torch.randn(1, in_channels, 16, 16),
            torch.randn(1, in_channels, 16, 16),
        ]
        
        output_dw = module_dw(x_list)
        output_std = module_std(x_list)
        
        # Both should produce same shape
        assert output_dw.shape == output_std.shape == (1, out_channels, 16, 16)

    def test_post_fusion_option(self):
        """Test with and without post-fusion processing."""
        in_channels_list = [64, 128]
        out_channels = 96
        
        # Without post-fusion
        module_no_post = NewConcat(in_channels_list, out_channels, post_fusion=False)
        
        # With post-fusion
        module_with_post = NewConcat(in_channels_list, out_channels, post_fusion=True)
        
        x_list = [
            torch.randn(1, 64, 16, 16),
            torch.randn(1, 128, 16, 16),
        ]
        
        output_no_post = module_no_post(x_list)
        output_with_post = module_with_post(x_list)
        
        # Both should produce same shape
        assert output_no_post.shape == output_with_post.shape == (1, out_channels, 16, 16)

    def test_single_input(self):
        """Test with single input (edge case)."""
        in_channels_list = [128]
        out_channels = 64
        module = NewConcat(in_channels_list, out_channels)
        
        x_list = [torch.randn(2, 128, 24, 24)]
        
        output = module(x_list)
        assert output.shape == (2, out_channels, 24, 24)

    def test_multiple_inputs(self):
        """Test with many inputs."""
        in_channels_list = [32, 64, 128, 256, 512]
        out_channels = 128
        module = NewConcat(in_channels_list, out_channels)
        
        x_list = [
            torch.randn(1, 32, 64, 64),
            torch.randn(1, 64, 32, 32),
            torch.randn(1, 128, 16, 16),
            torch.randn(1, 256, 8, 8),
            torch.randn(1, 512, 4, 4),
        ]
        
        output = module(x_list)
        # Should align to largest (64x64)
        assert output.shape == (1, out_channels, 64, 64)

    def test_incorrect_input_count(self):
        """Test error handling for incorrect input count."""
        in_channels_list = [64, 128]
        out_channels = 96
        module = NewConcat(in_channels_list, out_channels)
        
        # Provide wrong number of inputs
        x_list = [torch.randn(1, 64, 16, 16)]  # Only 1 input instead of 2
        
        with pytest.raises(AssertionError):
            module(x_list)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the module."""
        in_channels_list = [64, 128]
        out_channels = 96
        module = NewConcat(in_channels_list, out_channels)
        
        x_list = [
            torch.randn(1, 64, 16, 16, requires_grad=True),
            torch.randn(1, 128, 16, 16, requires_grad=True),
        ]
        
        output = module(x_list)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients were computed
        assert x_list[0].grad is not None, "Gradients not computed for first input"
        assert x_list[1].grad is not None, "Gradients not computed for second input"

    def test_batch_size_consistency(self):
        """Test with different batch sizes."""
        in_channels_list = [64, 128]
        out_channels = 96
        module = NewConcat(in_channels_list, out_channels)
        
        for batch_size in [1, 2, 4, 8]:
            x_list = [
                torch.randn(batch_size, 64, 16, 16),
                torch.randn(batch_size, 128, 16, 16),
            ]
            
            output = module(x_list)
            assert output.shape == (batch_size, out_channels, 16, 16)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
