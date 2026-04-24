"""Phase 4 OOD healing strategies.

Three strategies:
  - targeted_resample: mask high-energy token positions, resample via EqM or LLM.
  - simplex_project: project high-energy sequences to the valid manifold, recover nearest tokens.
  - beam_rerank: rerank beam candidates by joint LM log-prob minus energy penalty.
"""

from aitchinson_flow.healing.targeted_resample import TargetedResampleResult, TargetedResampler
from aitchinson_flow.healing.simplex_project import SimplexProjectResult, SimplexProjectHealer
from aitchinson_flow.healing.beam_rerank import BeamRerankResult, BeamRerankScorer

__all__ = [
    "TargetedResampleResult",
    "TargetedResampler",
    "SimplexProjectResult",
    "SimplexProjectHealer",
    "BeamRerankResult",
    "BeamRerankScorer",
]
