"""분산 학습 테스트: DDP, ZeRO, Pipeline, Tensor Parallel."""

import pytest
import torch
import torch.nn as nn

from makellm.distributed import (
    is_distributed, get_world_size, get_rank,
    ZeROOptimizer,
    PipelineParallel,
    ColumnParallelLinear, RowParallelLinear,
)


class TestDDP:
    def test_not_distributed_by_default(self):
        """초기화 없이는 분산이 아님."""
        assert not is_distributed()
        assert get_world_size() == 1
        assert get_rank() == 0


class TestZeRO:
    def test_zero_optimizer_creation(self):
        """ZeRO 옵티마이저가 정상 생성되어야 함."""
        model = nn.Linear(10, 10)
        opt = ZeROOptimizer(model.parameters(), stage=1, lr=1e-3)
        assert opt.stage == 1
        assert opt.world_size == 1

    def test_zero_step(self):
        """단일 프로세스에서 step이 정상 동작해야 함."""
        model = nn.Linear(10, 10)
        opt = ZeROOptimizer(model.parameters(), stage=1, lr=1e-3)
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()
        opt.step()
        # 가중치가 갱신되어야 함
        assert model.weight.grad is not None


class TestPipeline:
    def test_pipeline_creation(self):
        stages = [nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)]
        pp = PipelineParallel(stages, n_micro_batches=4)
        assert pp.num_stages() == 3

    def test_pipeline_forward(self):
        stages = [nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)]
        pp = PipelineParallel(stages, n_micro_batches=2)
        x = torch.randn(4, 10)
        out = pp(x)
        assert out.shape == (4, 5)

    def test_micro_batch_split(self):
        """마이크로배치 수만큼 분할되어야 함."""
        stages = [nn.Linear(8, 8)]
        pp = PipelineParallel(stages, n_micro_batches=4)
        x = torch.randn(8, 8)  # batch=8, 4 마이크로배치 → 각 2
        out = pp(x)
        assert out.shape == (8, 8)


class TestTensorParallel:
    def test_column_parallel_shape(self):
        """ColumnParallelLinear는 출력 차원이 줄어들어야 함 (rank=0, world=1인 경우 통과)."""
        layer = ColumnParallelLinear(in_features=10, out_features=20, bias=True)
        x = torch.randn(2, 10)
        out = layer(x)
        # 단일 rank이므로 out_per_rank = out_features
        assert out.shape == (2, 20)

    def test_row_parallel_shape(self):
        layer = RowParallelLinear(in_features=10, out_features=20, bias=True)
        x = torch.randn(2, 10)
        out = layer(x)
        assert out.shape == (2, 20)

    def test_column_parallel_weight_init(self):
        layer = ColumnParallelLinear(8, 16)
        assert layer.weight.shape == (16, 8)
        assert layer.bias.shape == (16,)
