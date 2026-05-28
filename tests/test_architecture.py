import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base import BaseSurvivalModel
from src.models.cox import CoxPHModel


def test_base_survival_model_is_abstract():
    try:
        BaseSurvivalModel()
    except TypeError:
        return

    raise AssertionError("BaseSurvivalModel should not be instantiable directly.")


def test_cox_model_implements_base_interface():
    cox = CoxPHModel(penalizer=0.1)

    assert isinstance(cox, BaseSurvivalModel)
    assert cox.penalizer == 0.1


if __name__ == "__main__":
    test_base_survival_model_is_abstract()
    test_cox_model_implements_base_interface()
