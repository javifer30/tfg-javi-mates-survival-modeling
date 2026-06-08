import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.coxph_tfg import train_coxph
from src.models.deephit_tfg import DeepHitNet, build_deephit_masks
from src.models.deepsurv_tfg import DeepSurvNet, cox_partial_likelihood_loss
from src.models.kaplan_meier_tfg import train_kaplan_meier
from src.models.pchazard_tfg import train_pchazard


def test_static_model_entrypoints_are_importable():
    assert callable(train_kaplan_meier)
    assert callable(train_coxph)
    assert callable(train_pchazard)


def test_deep_model_components_are_importable():
    assert DeepSurvNet is not None
    assert callable(cox_partial_likelihood_loss)
    assert DeepHitNet is not None
    assert callable(build_deephit_masks)


if __name__ == "__main__":
    test_static_model_entrypoints_are_importable()
    test_deep_model_components_are_importable()
