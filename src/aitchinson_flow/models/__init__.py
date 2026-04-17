from aitchinson_flow.models.base import (
    GenerativeTrainingModel,
    LossDict,
    TRAINING_LOSS_KEY,
    AuditorModel,
)
from aitchinson_flow.models.factory import (
    REGISTRY,
    build_model,
    register,
    registered_model_names,
)
from aitchinson_flow.models.bayesian_auditor import BayesianAuditor
from aitchinson_flow.models.bayesian_generator import BayesianGenerator
from aitchinson_flow.models.per_token_bayesian_auditor import PerTokenBayesianAuditor

# Populate REGISTRY (side-effect imports)
from aitchinson_flow.models import equilibrium  # noqa: F401
from aitchinson_flow.models import flow_matching  # noqa: F401
from aitchinson_flow.models import frozen_backbone  # noqa: F401
__all__ = [
    "GenerativeTrainingModel",
    "LossDict",
    "TRAINING_LOSS_KEY",
    "REGISTRY",
    "build_model",
    "register",
    "registered_model_names",
    "AuditorModel",
    "BayesianAuditor",
    "BayesianGenerator",
    "PerTokenBayesianAuditor",
]
