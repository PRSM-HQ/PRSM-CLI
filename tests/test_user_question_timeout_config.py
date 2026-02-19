from __future__ import annotations

import os
import tempfile
from pathlib import Path

from prsm.engine.config import EngineConfig
from prsm.engine.yaml_config import load_yaml_config


def test_engine_config_default_user_question_timeout_is_disabled() -> None:
    cfg = EngineConfig()
    assert cfg.user_question_timeout_seconds == 0.0


def test_engine_config_from_env_user_question_timeout() -> None:
    old_value = os.environ.get("ORCH_USER_QUESTION_TIMEOUT")
    os.environ["ORCH_USER_QUESTION_TIMEOUT"] = "5400"
    try:
        cfg = EngineConfig.from_env()
        assert cfg.user_question_timeout_seconds == 5400.0
    finally:
        if old_value is None:
            os.environ.pop("ORCH_USER_QUESTION_TIMEOUT", None)
        else:
            os.environ["ORCH_USER_QUESTION_TIMEOUT"] = old_value


def test_yaml_config_loads_user_question_timeout() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "prsm.yaml"
        config_path.write_text(
            "engine:\n"
            "  user_question_timeout_seconds: 4800\n"
        )
        cfg = load_yaml_config(config_path)
        assert cfg.engine.user_question_timeout_seconds == 4800.0
