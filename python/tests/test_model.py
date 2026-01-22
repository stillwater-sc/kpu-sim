# python/tests/test_model.py
"""
Tests for KPU Model-Level Execution (v0.8.0).

Tests:
- Model, Layer, Sequential classes
- Model loading (JSON, state dict)
- Inference pipeline
- Memory planning
- Reference models (SqueezeNet, MobileNetV2, MNIST)
"""

import numpy as np
import pytest
import tempfile
import json
from pathlib import Path

import kpu
from kpu.tensor import Tensor
from kpu.model import (
    Layer, Sequential, Model, Linear, Conv2d, BatchNorm2d, LayerNorm,
    ReLU, GELU, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Flatten, Dropout,
    save_state_dict, load_state_dict_from_file,
)
from kpu.model_loader import ModelLoader, load_model, save_model
from kpu.inference import InferencePipeline, run_inference, profile_model, benchmark_model
from kpu.memory_planner import MemoryPlanner, MemoryLevel, plan_memory, estimate_memory


class TestLayerClasses:
    """Test individual layer classes."""

    def test_linear_layer(self):
        """Test Linear layer."""
        layer = Linear(10, 5)

        assert layer.in_features == 10
        assert layer.out_features == 5
        assert "weight" in layer._parameters
        assert "bias" in layer._parameters

        # Forward pass
        x = Tensor(np.random.randn(2, 10).astype(np.float32))
        y = layer(x)

        assert y.shape == (2, 5)

    def test_linear_no_bias(self):
        """Test Linear layer without bias."""
        layer = Linear(10, 5, bias=False)

        assert layer.has_bias is False
        assert layer._parameters.get("bias") is None

    def test_conv2d_layer(self):
        """Test Conv2d layer."""
        layer = Conv2d(3, 16, kernel_size=3, padding=1)

        assert layer.in_channels == 3
        assert layer.out_channels == 16
        assert layer.kernel_size == (3, 3)
        assert layer.padding == (1, 1)

        # Forward pass
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        y = layer(x)

        assert y.shape == (1, 16, 32, 32)

    def test_conv2d_stride(self):
        """Test Conv2d with stride."""
        layer = Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        y = layer(x)

        assert y.shape == (1, 16, 16, 16)

    def test_batchnorm2d_layer(self):
        """Test BatchNorm2d layer."""
        layer = BatchNorm2d(16)

        assert layer.num_features == 16
        assert "weight" in layer._parameters
        assert "bias" in layer._parameters
        assert "running_mean" in layer._buffers
        assert "running_var" in layer._buffers

        # Forward pass
        x = Tensor(np.random.randn(2, 16, 8, 8).astype(np.float32))
        y = layer(x)

        assert y.shape == x.shape

    def test_activation_layers(self):
        """Test activation layers."""
        x = Tensor(np.array([[-1, 0, 1], [2, -2, 0]]).astype(np.float32))

        # ReLU
        relu = ReLU()
        y = relu(x)
        assert np.all(y.numpy() >= 0)

        # GELU
        gelu = GELU()
        y = gelu(x)
        assert y.shape == x.shape

    def test_pooling_layers(self):
        """Test pooling layers."""
        x = Tensor(np.random.randn(1, 3, 8, 8).astype(np.float32))

        # MaxPool2d
        maxpool = MaxPool2d(kernel_size=2, stride=2)
        y = maxpool(x)
        assert y.shape == (1, 3, 4, 4)

        # AvgPool2d
        avgpool = AvgPool2d(kernel_size=2, stride=2)
        y = avgpool(x)
        assert y.shape == (1, 3, 4, 4)

        # AdaptiveAvgPool2d
        adaptive = AdaptiveAvgPool2d((1, 1))
        y = adaptive(x)
        assert y.shape == (1, 3, 1, 1)

    def test_flatten_layer(self):
        """Test Flatten layer."""
        x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32))

        flatten = Flatten()
        y = flatten(x)

        assert y.shape == (2, 48)

    def test_dropout_layer(self):
        """Test Dropout layer (should be identity in inference)."""
        x = Tensor(np.random.randn(2, 10).astype(np.float32))

        dropout = Dropout(p=0.5)
        y = dropout(x)

        # In inference mode, dropout should be identity
        assert np.allclose(y.numpy(), x.numpy())


class TestSequential:
    """Test Sequential container."""

    def test_sequential_basic(self):
        """Test basic Sequential usage."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2),
        )

        assert len(model) == 3

        x = Tensor(np.random.randn(2, 10).astype(np.float32))
        y = model(x)

        assert y.shape == (2, 2)

    def test_sequential_indexing(self):
        """Test Sequential layer indexing."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2),
        )

        assert isinstance(model[0], Linear)
        assert isinstance(model[1], ReLU)
        assert isinstance(model[2], Linear)

    def test_sequential_iteration(self):
        """Test Sequential iteration."""
        layers = [Linear(10, 5), ReLU(), Linear(5, 2)]
        model = Sequential(*layers)

        for i, layer in enumerate(model):
            assert isinstance(layer, (Linear, ReLU))

    def test_sequential_append(self):
        """Test Sequential append."""
        model = Sequential(Linear(10, 5))
        model.append(ReLU())
        model.append(Linear(5, 2))

        assert len(model) == 3


class TestModelBase:
    """Test Model base class."""

    def test_model_parameters(self):
        """Test parameter iteration."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2),
        )

        params = list(model.parameters())
        # Should have weight and bias for each Linear
        assert len(params) == 4  # 2 weights + 2 biases

    def test_model_num_parameters(self):
        """Test parameter counting."""
        model = Sequential(
            Linear(10, 5, bias=False),  # 50 params
            ReLU(),
            Linear(5, 2, bias=False),   # 10 params
        )

        assert model.num_parameters() == 60

    def test_model_state_dict(self):
        """Test state dict save/load."""
        model1 = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2),
        )

        # Get state dict
        state = model1.state_dict()
        assert "layer0.weight" in state
        assert "layer0.bias" in state

        # Create new model and load
        model2 = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2),
        )
        model2.load_state_dict(state)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model1.parameters(), model2.parameters()):
            assert np.allclose(p1.numpy(), p2.numpy())

    def test_model_save_load_file(self):
        """Test saving/loading state dict to file."""
        model = Sequential(
            Linear(10, 5),
            Linear(5, 2),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.npz"

            # Save
            save_state_dict(model, str(path))
            assert path.exists()

            # Load
            state = load_state_dict_from_file(str(path))
            assert "layer0.weight" in state

    def test_model_summary(self):
        """Test model summary."""
        model = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 10),
        )

        summary = model.summary()
        assert "layer0" in summary
        assert "layer2" in summary


class TestModelLoader:
    """Test ModelLoader class."""

    def test_load_from_json(self):
        """Test loading model from JSON."""
        model_def = {
            "name": "test_model",
            "version": "1.0",
            "inputs": ["input"],
            "outputs": ["output"],
            "layers": [
                {"type": "linear", "name": "fc1", "params": {"in_features": 10, "out_features": 5}},
                {"type": "relu", "name": "relu1"},
                {"type": "linear", "name": "fc2", "params": {"in_features": 5, "out_features": 2}},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            with open(path, "w") as f:
                json.dump(model_def, f)

            model = ModelLoader.from_json(path)

            assert model is not None
            x = Tensor(np.random.randn(2, 10).astype(np.float32))
            y = model(x)
            assert y.shape == (2, 2)

    def test_save_to_json(self):
        """Test saving model to JSON."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2),
            name="test_model",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"

            ModelLoader.save_to_json(model, path)

            assert path.exists()
            assert path.with_suffix(".npz").exists()

            # Load back
            with open(path) as f:
                data = json.load(f)
            assert data["name"] == "test_model"
            assert len(data["layers"]) == 3

    def test_load_model_auto_format(self):
        """Test auto-format detection in load_model."""
        model_def = {
            "name": "auto_test",
            "layers": [
                {"type": "linear", "params": {"in_features": 10, "out_features": 5}},
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            with open(path, "w") as f:
                json.dump(model_def, f)

            model = load_model(path)
            assert model is not None


class TestInferencePipeline:
    """Test InferencePipeline class."""

    def test_basic_inference(self):
        """Test basic inference pipeline."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2),
        )

        pipeline = InferencePipeline(model)

        x = Tensor(np.random.randn(4, 10).astype(np.float32))
        y = pipeline(x)

        assert y.shape == (4, 2)

    def test_inference_with_stats(self):
        """Test inference with statistics."""
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2),
        )

        pipeline = InferencePipeline(model, profile_layers=True)

        x = Tensor(np.random.randn(4, 10).astype(np.float32))
        y, stats = pipeline(x, return_stats=True)

        assert y.shape == (4, 2)
        assert stats is not None
        assert stats.batch_size == 4
        assert stats.total_execution_time_ms > 0

    def test_run_inference_helper(self):
        """Test run_inference helper function."""
        model = Sequential(
            Linear(10, 5),
            Linear(5, 2),
        )

        x = Tensor(np.random.randn(2, 10).astype(np.float32))
        y = run_inference(model, x)

        assert y.shape == (2, 2)

    def test_profile_model_helper(self):
        """Test profile_model helper function."""
        model = Sequential(
            Linear(10, 5),
            Linear(5, 2),
        )

        stats = profile_model(model, input_shape=(2, 10))

        assert stats is not None
        assert stats.batch_size == 2


class TestMemoryPlanner:
    """Test MemoryPlanner class."""

    def test_basic_planning(self):
        """Test basic memory planning."""
        model = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 10),
        )

        planner = MemoryPlanner(model)
        plan = planner.plan(input_shape=(1, 784))

        assert plan is not None
        assert plan.total_tensor_bytes > 0
        assert plan.peak_memory_bytes > 0

    def test_plan_memory_helper(self):
        """Test plan_memory helper function."""
        model = Sequential(
            Linear(100, 50),
            Linear(50, 10),
        )

        plan = plan_memory(model, input_shape=(1, 100))

        assert plan.total_tensor_bytes > 0

    def test_estimate_memory_helper(self):
        """Test estimate_memory helper function."""
        model = Sequential(
            Linear(100, 50),
            Linear(50, 10),
        )

        estimates = estimate_memory(model, input_shape=(1, 100))

        assert "total_bytes" in estimates
        assert "peak_bytes" in estimates
        assert "weight_bytes" in estimates

    def test_memory_levels(self):
        """Test MemoryLevel enum."""
        assert MemoryLevel.DRAM.value < MemoryLevel.L3.value
        assert MemoryLevel.L3.value < MemoryLevel.L2.value

        # Check capacities
        assert MemoryLevel.DRAM.capacity_bytes > MemoryLevel.L3.capacity_bytes
        assert MemoryLevel.L3.capacity_bytes > MemoryLevel.L2.capacity_bytes


class TestReferenceModels:
    """Test reference model implementations."""

    def test_mnist_mlp(self):
        """Test MNIST MLP model."""
        from kpu.models import mnist_mlp

        model = mnist_mlp()

        assert model.input_size == 784
        assert model.output_size == 10

        # Forward pass
        x = Tensor(np.random.randn(2, 784).astype(np.float32))
        y = model(x)

        assert y.shape == (2, 10)

    def test_mnist_cnn(self):
        """Test MNIST CNN model."""
        from kpu.models import mnist_cnn

        model = mnist_cnn()

        assert model.in_channels == 1
        assert model.num_classes == 10

        # Forward pass
        x = Tensor(np.random.randn(2, 1, 28, 28).astype(np.float32))
        y = model(x)

        assert y.shape == (2, 10)

    def test_squeezenet_structure(self):
        """Test SqueezeNet model structure."""
        from kpu.models import squeezenet1_0, Fire

        model = squeezenet1_0(num_classes=10)

        assert model.num_classes == 10
        assert model.version == "1_0"

        # Check Fire module exists
        assert "fire2" in model._children
        fire = model._children["fire2"]
        assert isinstance(fire, Fire)

    def test_squeezenet_forward(self):
        """Test SqueezeNet forward pass."""
        from kpu.models import squeezenet1_0

        model = squeezenet1_0(num_classes=10)

        # Forward pass with smaller input (64x64) for faster testing
        # Full 224x224 works but is slow in behavioral mode
        x = Tensor(np.random.randn(1, 3, 64, 64).astype(np.float32))
        y = model(x)

        assert y.shape == (1, 10)

    def test_squeezenet_1_1(self):
        """Test SqueezeNet 1.1 variant."""
        from kpu.models import squeezenet1_1

        model = squeezenet1_1(num_classes=100)

        assert model.version == "1_1"
        assert model.num_classes == 100

        # Use smaller input for faster testing
        x = Tensor(np.random.randn(1, 3, 64, 64).astype(np.float32))
        y = model(x)

        assert y.shape == (1, 100)

    def test_mobilenet_structure(self):
        """Test MobileNetV2 model structure."""
        from kpu.models import mobilenet_v2, InvertedResidual

        model = mobilenet_v2(num_classes=10)

        assert model.num_classes == 10

        # Check InvertedResidual exists
        assert "block0" in model._children
        block = model._children["block0"]
        assert isinstance(block, InvertedResidual)

    @pytest.mark.slow
    def test_mobilenet_forward(self):
        """Test MobileNetV2 forward pass.

        Note: MobileNetV2 has many layers and is slow in behavioral mode.
        This test is marked as slow and uses a small input (32x32).
        """
        from kpu.models import mobilenet_v2

        model = mobilenet_v2(num_classes=10)

        # Use very small input for faster testing (MobileNet has many layers)
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        y = model(x)

        assert y.shape == (1, 10)


class TestIntegration:
    """Integration tests for model-level execution."""

    def test_end_to_end_mlp(self):
        """Test end-to-end MLP execution."""
        from kpu.models import mnist_mlp

        # Create model
        model = mnist_mlp(hidden_sizes=[64, 32])

        # Create pipeline
        pipeline = InferencePipeline(model, profile_layers=True)

        # Run inference
        x = Tensor(np.random.randn(32, 784).astype(np.float32))
        output, stats = pipeline(x, return_stats=True)

        # Verify
        assert output.shape == (32, 10)
        assert stats.batch_size == 32
        assert len(stats.layer_stats) > 0

    def test_end_to_end_cnn(self):
        """Test end-to-end CNN execution."""
        from kpu.models import mnist_cnn

        # Create model
        model = mnist_cnn()

        # Create pipeline
        pipeline = InferencePipeline(model)

        # Run inference
        x = Tensor(np.random.randn(8, 1, 28, 28).astype(np.float32))
        output = pipeline(x)

        assert output.shape == (8, 10)

    def test_memory_planning_with_inference(self):
        """Test memory planning combined with inference."""
        from kpu.models import mnist_mlp

        model = mnist_mlp()

        # Plan memory
        plan = plan_memory(model, input_shape=(1, 784))

        # Run inference
        pipeline = InferencePipeline(model)
        x = Tensor(np.random.randn(1, 784).astype(np.float32))
        output = pipeline(x)

        # Both should work
        assert plan.peak_memory_bytes > 0
        assert output.shape == (1, 10)

    def test_save_load_roundtrip(self):
        """Test model save/load roundtrip."""
        from kpu.models import mnist_mlp

        # Create and save model
        model1 = mnist_mlp(hidden_sizes=[32])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            save_model(model1, path)

            # Load model
            model2 = load_model(path)

            # Both should produce same output structure
            x = Tensor(np.random.randn(2, 784).astype(np.float32))
            y1 = model1(x)
            y2 = model2(x)

            assert y1.shape == y2.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
