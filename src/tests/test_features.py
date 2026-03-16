import pandas as pd
import numpy as np
import pytest
from src.features.builder import FeatureBuilder

def test_target_scaling_roundtrip(tmp_path, monkeypatch):
    # use temporary config to avoid interfering with real models_dir
    cfg = {
        "data": {"target_col": "Close", "sequence_length": 24},
        "paths": {"models_dir": str(tmp_path)}
    }
    # write temp config
    cfg_path = tmp_path / "config.yaml"
    import yaml
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    fb = FeatureBuilder(str(cfg_path))

    # create a simple series of values
    series = pd.Series([100.0, 200.0, 300.0], index=[0,1,2])
    scaled = fb.scale_target(series, is_training=True)
    assert scaled.mean() == pytest.approx(0.0, abs=1e-6)

    inv = fb.inverse_scale_target(scaled)
    assert np.allclose(inv, series.values)
