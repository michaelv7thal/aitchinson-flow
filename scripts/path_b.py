import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "two_stage_train", Path(__file__).parent / "two_stage_train.py"
)
two_stage_train = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules["two_stage_train"] = two_stage_train
spec.loader.exec_module(two_stage_train)  # type: ignore[union-attr]
run_two_stage = two_stage_train.run_two_stage

from aitchinson_flow.config import Config, LLMEmbeddingDatasetConfig

cfg = Config()
cfg.training_data.source = "llm_topk"
cfg.teacher.model_id = "Qwen/Qwen2.5-0.5B"  # or "gpt2"
cfg.teacher.dtype = "bfloat16"
cfg.llm_embedding_dataset = LLMEmbeddingDatasetConfig(
    char_window_length=256,
    corrupt_rate=0.15,
)
cfg.dataset.K = 64  # simplex dim of the learned TokenEmbeddingToSimplex projection
cfg.dataset.L = 30  # LLM-token length

run_two_stage(cfg, out_dir=Path("checkpoints/two_stage/path_b"), stage1_epochs=30, stage2_epochs=30)
