"""Tests for model alias resolution in ModelRegistry and yaml_config."""
import os
import tempfile

import pytest
import yaml

from prsm.engine.model_registry import (
    ModelCapability,
    ModelRegistry,
    ModelTier,
    build_default_registry,
    load_model_registry_from_yaml,
)
from prsm.engine.yaml_config import (
    ModelAlias,
    load_yaml_config,
    resolve_model_alias,
    _BUILTIN_CLAUDE_ALIASES,
)
from prsm.vscode.server import PrsmServer


# ── ModelRegistry alias tests ───────────────────────────────────


class TestRegistryAliases:
    """Test alias resolution in ModelRegistry."""

    def test_builtin_claude_aliases_populated(self):
        """Built-in Claude family aliases are available immediately."""
        reg = ModelRegistry()
        aliases = reg.list_aliases()
        assert "claude-sonnet" in aliases
        assert "claude-opus" in aliases
        assert "claude-haiku" in aliases
        assert "sonnet" in aliases
        assert "opus" in aliases
        assert "haiku" in aliases

    def test_resolve_alias_direct_model_id(self):
        """When name is a registered model_id, return it as-is."""
        reg = build_default_registry()
        assert reg.resolve_alias("claude-opus-4-6") == "claude-opus-4-6"
        assert reg.resolve_alias("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5-20250929"

    def test_resolve_alias_short_names(self):
        """Short names resolve to full versioned IDs."""
        reg = build_default_registry()
        assert reg.resolve_alias("claude-sonnet") == "claude-sonnet-4-5-20250929"
        assert reg.resolve_alias("claude-opus") == "claude-opus-4-6"
        assert reg.resolve_alias("claude-haiku") == "claude-3-5-haiku-20241022"

    def test_resolve_alias_even_shorter_names(self):
        """Even shorter names (sonnet, opus, haiku) also resolve."""
        reg = build_default_registry()
        assert reg.resolve_alias("sonnet") == "claude-sonnet-4-5-20250929"
        assert reg.resolve_alias("opus") == "claude-opus-4-6"
        assert reg.resolve_alias("haiku") == "claude-3-5-haiku-20241022"

    def test_resolve_alias_unknown_passthrough(self):
        """Unknown names pass through unchanged."""
        reg = build_default_registry()
        assert reg.resolve_alias("some-random-model") == "some-random-model"

    def test_get_resolves_aliases(self):
        """registry.get() transparently resolves aliases."""
        reg = build_default_registry()
        cap = reg.get("claude-sonnet")
        assert cap is not None
        assert cap.model_id == "claude-sonnet-4-5-20250929"
        assert cap.provider == "claude"

    def test_get_by_short_name(self):
        """registry.get('opus') returns the right model."""
        reg = build_default_registry()
        cap = reg.get("opus")
        assert cap is not None
        assert cap.model_id == "claude-opus-4-6"
        assert cap.tier == ModelTier.FRONTIER

    def test_get_runtime_variant_exact_match(self):
        """Runtime-encoded model IDs should resolve to exact entries."""
        reg = ModelRegistry()
        reg.register(ModelCapability(
            model_id="gpt-5-3::reasoning_effort=high",
            provider="codex",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))

        cap = reg.get("gpt-5-3::reasoning_effort=high")
        assert cap is not None
        assert cap.model_id == "gpt-5-3::reasoning_effort=high"
        assert cap.provider == "codex"

    def test_is_model_available_with_alias(self):
        """is_model_available works with aliases."""
        reg = build_default_registry()
        # All Claude models are available by default in the default registry
        assert reg.is_model_available("claude-sonnet") is True
        assert reg.is_model_available("opus") is True

    def test_is_model_available_runtime_variant_exact_match(self):
        """Availability checks should support runtime-encoded model IDs."""
        reg = ModelRegistry()
        reg.register(ModelCapability(
            model_id="gpt-5-3::reasoning_effort=high",
            provider="codex",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))
        assert reg.is_model_available("gpt-5-3::reasoning_effort=high") is True

    def test_resolve_alias_with_provider(self):
        """resolve_alias_with_provider returns model_id and provider."""
        reg = build_default_registry()
        model_id, provider = reg.resolve_alias_with_provider("claude-sonnet")
        assert model_id == "claude-sonnet-4-5-20250929"
        assert provider == "claude"

    def test_resolve_alias_with_provider_runtime_variant_exact_match(self):
        """Provider resolution should preserve runtime-encoded variants."""
        reg = ModelRegistry()
        reg.register(ModelCapability(
            model_id="gpt-5-3::reasoning_effort=high",
            provider="codex",
            tier=ModelTier.FRONTIER,
            cost_factor=1.0,
            speed_factor=1.0,
            max_context=200_000,
            affinities={},
        ))
        model_id, provider = reg.resolve_alias_with_provider("gpt-5-3::reasoning_effort=high")
        assert model_id == "gpt-5-3::reasoning_effort=high"
        assert provider == "codex"

    def test_resolve_alias_with_provider_unknown(self):
        """resolve_alias_with_provider returns None provider for unknown models."""
        reg = build_default_registry()
        model_id, provider = reg.resolve_alias_with_provider("unknown-model")
        assert model_id == "unknown-model"
        assert provider is None

    def test_register_custom_alias(self):
        """Custom aliases can be registered and resolved."""
        reg = build_default_registry()
        reg.register_alias("my-fast-model", "claude-3-5-haiku-20241022")
        assert reg.resolve_alias("my-fast-model") == "claude-3-5-haiku-20241022"
        cap = reg.get("my-fast-model")
        assert cap is not None
        assert cap.model_id == "claude-3-5-haiku-20241022"

    def test_yaml_model_aliases_registered(self):
        """YAML model aliases are registered during load_model_registry_from_yaml."""
        yaml_models = {
            "my-sonnet": ModelAlias(provider="claude", model_id="claude-sonnet-4-5-20250929"),
            "gpt": ModelAlias(provider="codex", model_id="gpt-5.2-codex"),
        }
        reg = load_model_registry_from_yaml(
            {},  # No registry overrides
            build_default_registry(),
            model_aliases=yaml_models,
        )
        assert reg.resolve_alias("my-sonnet") == "claude-sonnet-4-5-20250929"
        assert reg.resolve_alias("gpt") == "gpt-5.2-codex"

    def test_yaml_dict_aliases_registered(self):
        """YAML model aliases as dicts (not ModelAlias objects) work too."""
        yaml_models = {
            "fast": {"provider": "claude", "model_id": "claude-3-5-haiku-20241022"},
        }
        reg = load_model_registry_from_yaml(
            {}, build_default_registry(),
            model_aliases=yaml_models,
        )
        assert reg.resolve_alias("fast") == "claude-3-5-haiku-20241022"

    def test_to_summary_includes_aliases(self):
        """to_summary() shows alias information."""
        reg = build_default_registry()
        summary = reg.to_summary()
        assert "aliases:" in summary
        assert "claude-sonnet" in summary
        assert "claude-opus" in summary


# ── yaml_config alias tests ──────────────────────────────────────


class TestYamlConfigAliases:
    """Test resolve_model_alias with built-in Claude family aliases."""

    def test_yaml_alias_takes_priority(self):
        """YAML-defined aliases take priority over built-in ones."""
        models = {
            "claude-sonnet": ModelAlias(
                provider="custom-provider",
                model_id="custom-sonnet-v99",
            ),
        }
        provider, model_id = resolve_model_alias("claude-sonnet", models)
        assert provider == "custom-provider"
        assert model_id == "custom-sonnet-v99"

    def test_builtin_alias_fallback(self):
        """Built-in aliases work when not overridden in YAML."""
        models = {}  # Empty YAML models
        provider, model_id = resolve_model_alias("claude-sonnet", models)
        assert provider == "claude"
        assert model_id == "claude-sonnet-4-5-20250929"

    def test_builtin_alias_claude_opus(self):
        provider, model_id = resolve_model_alias("claude-opus", {})
        assert provider == "claude"
        assert model_id == "claude-opus-4-6"

    def test_builtin_alias_claude_haiku(self):
        provider, model_id = resolve_model_alias("claude-haiku", {})
        assert provider == "claude"
        assert model_id == "claude-3-5-haiku-20241022"

    def test_raw_model_id_passthrough(self):
        """Unknown names are treated as raw model IDs with claude provider."""
        provider, model_id = resolve_model_alias("gpt-4o-mini", {})
        assert provider == "claude"
        assert model_id == "gpt-4o-mini"

    def test_builtin_aliases_match_registry(self):
        """Built-in aliases in yaml_config match those in ModelRegistry."""
        from prsm.engine.model_registry import ModelRegistry
        reg_aliases = ModelRegistry.CLAUDE_FAMILY_ALIASES
        for alias, (provider, model_id) in _BUILTIN_CLAUDE_ALIASES.items():
            assert alias in reg_aliases, f"yaml alias {alias} missing from registry"
            assert reg_aliases[alias] == model_id, (
                f"yaml alias {alias} maps to {model_id} but registry has {reg_aliases[alias]}"
            )


# ── global models.yaml merging tests ─────────────────────────────


class TestModelsYamlMerge:
    """Test that load_yaml_config merges global ~/.prsm/models.yaml."""

    def _write_yaml(self, path, data):
        with open(path, "w") as f:
            yaml.safe_dump(data, f)

    @staticmethod
    def _global_models_path(home_dir: str) -> str:
        return os.path.join(home_dir, ".prsm", "models.yaml")

    def test_models_yaml_aliases_loaded(self, monkeypatch):
        """Aliases defined in ~/.prsm/models.yaml are available after load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("HOME", tmpdir)
            models_yaml = self._global_models_path(tmpdir)
            os.makedirs(os.path.dirname(models_yaml), exist_ok=True)
            self._write_yaml(models_yaml, {
                "models": {
                    "gpt-5-3-low": {
                        "provider": "codex",
                        "model_id": "gpt-5-3",
                        "reasoning_effort": "low",
                    },
                    "gpt-5-3-medium": {
                        "provider": "codex",
                        "model_id": "gpt-5-3",
                        "reasoning_effort": "medium",
                    },
                    "gpt-5-3-high": {
                        "provider": "codex",
                        "model_id": "gpt-5-3",
                        "reasoning_effort": "high",
                    },
                },
            })
            self._write_yaml(os.path.join(tmpdir, "prsm.yaml"), {
                "providers": {"codex": {"type": "codex", "command": "codex"}},
            })

            cfg = load_yaml_config(os.path.join(tmpdir, "prsm.yaml"))

            assert "gpt-5-3-low" in cfg.models
            assert "gpt-5-3-medium" in cfg.models
            assert "gpt-5-3-high" in cfg.models
            assert cfg.models["gpt-5-3-low"].reasoning_effort == "low"
            assert cfg.models["gpt-5-3-medium"].reasoning_effort == "medium"
            assert cfg.models["gpt-5-3-high"].reasoning_effort == "high"
            for alias in ("gpt-5-3-low", "gpt-5-3-medium", "gpt-5-3-high"):
                assert cfg.models[alias].provider == "codex"
                assert cfg.models[alias].model_id == "gpt-5-3"

    def test_prsm_yaml_overrides_models_yaml(self, monkeypatch):
        """prsm.yaml models take precedence over ~/.prsm/models.yaml on conflict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("HOME", tmpdir)
            models_yaml = self._global_models_path(tmpdir)
            os.makedirs(os.path.dirname(models_yaml), exist_ok=True)
            self._write_yaml(models_yaml, {
                "models": {
                    "my-model": {
                        "provider": "codex",
                        "model_id": "old-model",
                    },
                },
            })
            self._write_yaml(os.path.join(tmpdir, "prsm.yaml"), {
                "models": {
                    "my-model": {
                        "provider": "claude",
                        "model_id": "new-model",
                    },
                },
            })

            cfg = load_yaml_config(os.path.join(tmpdir, "prsm.yaml"))
            assert cfg.models["my-model"].provider == "claude"
            assert cfg.models["my-model"].model_id == "new-model"

    def test_model_registry_merged_from_models_yaml(self, monkeypatch):
        """model_registry from ~/.prsm/models.yaml is merged; prsm.yaml wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("HOME", tmpdir)
            models_yaml = self._global_models_path(tmpdir)
            os.makedirs(os.path.dirname(models_yaml), exist_ok=True)
            self._write_yaml(models_yaml, {
                "model_registry": {
                    "model-a": {"tier": "fast", "provider": "codex"},
                    "model-b": {"tier": "strong", "provider": "codex"},
                },
            })
            self._write_yaml(os.path.join(tmpdir, "prsm.yaml"), {
                "model_registry": {
                    "model-b": {"tier": "frontier", "provider": "claude"},
                },
            })

            cfg = load_yaml_config(os.path.join(tmpdir, "prsm.yaml"))
            assert "model-a" in cfg.model_registry_raw
            assert cfg.model_registry_raw["model-a"]["tier"] == "fast"
            assert cfg.model_registry_raw["model-b"]["tier"] == "frontier"

    def test_no_models_yaml_still_works(self, monkeypatch):
        """When ~/.prsm/models.yaml is missing, load_yaml_config still works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("HOME", tmpdir)
            self._write_yaml(os.path.join(tmpdir, "prsm.yaml"), {
                "models": {
                    "opus": {
                        "provider": "claude",
                        "model_id": "claude-opus-4-6",
                    },
                },
            })

            cfg = load_yaml_config(os.path.join(tmpdir, "prsm.yaml"))
            assert "opus" in cfg.models
            assert cfg.models["opus"].model_id == "claude-opus-4-6"

    def test_resolve_alias_with_merged_models(self, monkeypatch):
        """Aliases from ~/.prsm/models.yaml resolve via resolve_model_alias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("HOME", tmpdir)
            models_yaml = self._global_models_path(tmpdir)
            os.makedirs(os.path.dirname(models_yaml), exist_ok=True)
            self._write_yaml(models_yaml, {
                "models": {
                    "gpt-5-3-low": {
                        "provider": "codex",
                        "model_id": "gpt-5-3",
                        "reasoning_effort": "low",
                    },
                },
            })
            self._write_yaml(os.path.join(tmpdir, "prsm.yaml"), {})

            cfg = load_yaml_config(os.path.join(tmpdir, "prsm.yaml"))
            provider, model_id = resolve_model_alias("gpt-5-3-low", cfg.models)
            assert provider == "codex"
            assert model_id == "gpt-5-3::reasoning_effort=low"

    def test_real_project_models_yaml_loaded(self, monkeypatch):
        """Project prsm.yaml merges aliases from global ~/.prsm/models.yaml."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prsm_yaml = os.path.join(project_root, ".prism", "prsm.yaml")
        if not os.path.exists(prsm_yaml):
            prsm_yaml = os.path.join(project_root, "prsm.yaml")
        if not os.path.exists(prsm_yaml):
            pytest.skip("prsm.yaml not found at project root or .prism/")

        project_models_yaml = os.path.join(project_root, ".prism", "models.yaml")
        if not os.path.exists(project_models_yaml):
            pytest.skip(".prism/models.yaml template not found in project")

        with tempfile.TemporaryDirectory() as tmp_home:
            monkeypatch.setenv("HOME", tmp_home)
            global_models_yaml = self._global_models_path(tmp_home)
            os.makedirs(os.path.dirname(global_models_yaml), exist_ok=True)
            with open(project_models_yaml) as f:
                project_models_doc = yaml.safe_load(f) or {}
            self._write_yaml(global_models_yaml, project_models_doc)

            cfg = load_yaml_config(prsm_yaml)

            for alias in ("gpt-5-3-low", "gpt-5-3-medium", "gpt-5-3-high"):
                assert alias in cfg.models, f"{alias} not found in merged models"
                assert cfg.models[alias].provider == "codex"
                assert cfg.models[alias].model_id == "gpt-5-3"

            provider, mid = resolve_model_alias("gpt-5-3-low", cfg.models)
            assert provider == "codex"
            assert mid == "gpt-5-3::reasoning_effort=low"


class TestVscodeSettingsConfigSplit:
    """Test VS Code settings config handlers with global ~/.prsm/models.yaml."""

    def _write_yaml(self, path, data):
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @staticmethod
    def _global_models_path(home_dir: str) -> str:
        return os.path.join(home_dir, ".prsm", "models.yaml")

    def test_load_merged_config_includes_global_models_yaml(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("HOME", tmpdir)
            prsm_yaml = os.path.join(tmpdir, "prsm.yaml")
            models_yaml = self._global_models_path(tmpdir)
            os.makedirs(os.path.dirname(models_yaml), exist_ok=True)
            self._write_yaml(prsm_yaml, {
                "defaults": {
                    "model": "gpt-5-3-medium",
                    "peer_models": ["gpt-5-3-medium", "opus-4-6"],
                },
                "providers": {"codex": {"type": "codex"}},
            })
            self._write_yaml(models_yaml, {
                "models": {
                    "gpt-5-3-medium": {
                        "provider": "codex",
                        "model_id": "gpt-5-3",
                        "reasoning_effort": "medium",
                    },
                    "opus-4-6": {
                        "provider": "claude",
                        "model_id": "claude-opus-4-6",
                    },
                },
            })

            server = object.__new__(PrsmServer)
            merged = server._load_merged_config(prsm_yaml)

            assert merged["defaults"]["model"] == "gpt-5-3-medium"
            assert "models" in merged
            assert "gpt-5-3-medium" in merged["models"]
            assert merged["models"]["gpt-5-3-medium"]["reasoning_effort"] == "medium"
            assert "opus-4-6" in merged["models"]

    def test_write_split_config_writes_global_models_yaml_and_keeps_prsm_clean(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("HOME", tmpdir)
            prsm_yaml = os.path.join(tmpdir, "prsm.yaml")
            models_yaml = self._global_models_path(tmpdir)
            server = object.__new__(PrsmServer)

            new_config = {
                "engine": {"max_agent_depth": 5},
                "defaults": {
                    "model": "gpt-5-3-medium",
                    "peer_models": ["gpt-5-3-medium", "opus-4-6"],
                },
                "providers": {"codex": {"type": "codex"}},
                "models": {
                    "gpt-5-3-medium": {
                        "provider": "codex",
                        "model_id": "gpt-5-3",
                        "reasoning_effort": "medium",
                    },
                },
                "model_registry": {
                    "gpt-5-3::reasoning_effort=medium": {
                        "tier": "strong",
                        "provider": "codex",
                    },
                },
            }

            server._write_split_config(prsm_yaml, new_config)

            with open(prsm_yaml) as f:
                prsm_raw = yaml.safe_load(f) or {}
            with open(models_yaml) as f:
                models_raw = yaml.safe_load(f) or {}

            assert "models" not in prsm_raw
            assert "model_registry" not in prsm_raw
            assert prsm_raw["defaults"]["model"] == "gpt-5-3-medium"
            assert "models" in models_raw
            assert "gpt-5-3-medium" in models_raw["models"]
            assert "model_registry" in models_raw

    def test_resolve_config_path_prefers_project_yaml_for_generated_temp_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prism_dir = os.path.join(tmpdir, ".prism")
            os.makedirs(prism_dir, exist_ok=True)
            prsm_yaml = os.path.join(prism_dir, "prsm.yaml")
            self._write_yaml(prsm_yaml, {"defaults": {"model": "gpt-5-3-medium"}})

            server = object.__new__(PrsmServer)
            server._cwd = tmpdir
            server._config_path = os.path.join(tempfile.gettempdir(), "prsm-config.yaml")

            assert server._resolve_config_path() == prsm_yaml

    def test_load_merged_config_uses_global_models_yaml_for_temp_config(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as other_tmp:
            monkeypatch.setenv("HOME", tmpdir)
            models_yaml = self._global_models_path(tmpdir)
            temp_cfg = os.path.join(other_tmp, "prsm-config.yaml")
            os.makedirs(os.path.dirname(models_yaml), exist_ok=True)

            self._write_yaml(models_yaml, {
                "models": {
                    "gpt-5-3-medium": {
                        "provider": "codex",
                        "model_id": "gpt-5-3",
                        "reasoning_effort": "medium",
                    },
                },
                "model_registry": {
                    "gpt-5-3::reasoning_effort=medium": {
                        "tier": "strong",
                        "provider": "codex",
                        "affinities": {"coding": 0.89},
                    },
                },
            })
            self._write_yaml(temp_cfg, {"engine": {"max_agent_depth": 5}})

            server = object.__new__(PrsmServer)
            server._cwd = tmpdir

            merged = server._load_merged_config(temp_cfg)
            assert "gpt-5-3-medium" in merged.get("models", {})
            assert "gpt-5-3::reasoning_effort=medium" in merged.get("model_registry", {})

    def test_runtime_info_includes_affinities(self):
        server = object.__new__(PrsmServer)
        server._provider_registry = None
        server._yaml_config = None
        server._model_registry = ModelRegistry()
        server._model_registry.register(
            ModelCapability(
                model_id="unit-test-model",
                provider="codex",
                tier=ModelTier.STRONG,
                cost_factor=1.0,
                speed_factor=1.0,
                affinities={"coding": 0.91},
                available=True,
            )
        )

        rt = server._get_runtime_info()
        assert "unit-test-model" in rt["models"]
        assert rt["models"]["unit-test-model"]["affinities"]["coding"] == 0.91

    def test_load_merged_config_derives_registry_from_models_capability_fields(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("HOME", tmpdir)
            prsm_yaml = os.path.join(tmpdir, "prsm.yaml")
            models_yaml = self._global_models_path(tmpdir)
            os.makedirs(os.path.dirname(models_yaml), exist_ok=True)
            self._write_yaml(prsm_yaml, {"defaults": {"model": "gpt-5-3-medium"}})
            self._write_yaml(models_yaml, {
                "models": {
                    "gpt-5-3-medium": {
                        "provider": "codex",
                        "model_id": "gpt-5-3",
                        "reasoning_effort": "medium",
                        "tier": "strong",
                        "cost_factor": 1.2,
                        "speed_factor": 1.0,
                        "affinities": {"coding": 0.89},
                    },
                },
            })

            server = object.__new__(PrsmServer)
            server._cwd = tmpdir
            merged = server._load_merged_config(prsm_yaml)

            mr = merged.get("model_registry", {})
            key = "gpt-5-3::reasoning_effort=medium"
            assert key in mr
            assert mr[key]["tier"] == "strong"
            assert mr[key]["affinities"]["coding"] == 0.89
