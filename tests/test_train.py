import pandas as pd

from src.train import infer_synthetic_flag


def test_infer_synthetic_flag_handles_demo_and_external_data():
    assert infer_synthetic_flag(pd.DataFrame({"is_synthetic": [True, "true", "1"]}))
    assert not infer_synthetic_flag(pd.DataFrame({"message": ["hello"], "intent": ["billing"]}))
