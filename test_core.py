#!/usr/bin/env python3
"""
Test script for the text classification core
"""

from core import TextClassifier, TextInferConfig


def test_core_structure():
    """Test that core components can be imported and initialized"""
    print("Testing core structure...")

    # Test config initialization
    try:
        cfg = TextInferConfig()
        print("✓ TextInferConfig can be initialized")
    except Exception as e:
        print(f"✗ TextInferConfig initialization failed: {e}")
        return False

    # Test classifier initialization
    try:
        classifier = TextClassifier(cfg = cfg)
        print("✓ TextClassifier can be initialized")
    except Exception as e:
        print(f"✗ TextClassifier initialization failed: {e}")
        return False

    # Test public methods exist
    try:
        assert callable(classifier.load_models)
        assert callable(classifier.are_models_loaded)
        assert callable(classifier.classify_texts_batch)
        assert callable(classifier.get_model_info)
        print("✓ Core methods are callable")
    except Exception as e:
        print(f"✗ Core method check failed: {e}")
        return False

    print("✓ All core structure tests passed!")
    return True


def test_default_config_values():
    """Test default config values and types"""
    print("\nTesting default config values...")

    try:
        cfg = TextInferConfig()

        assert isinstance(cfg.repo_id, str)
        assert isinstance(cfg.subdir_by_lang_json, str)
        assert cfg.revision is None or isinstance(cfg.revision, str)
        assert cfg.hf_token is None or isinstance(cfg.hf_token, str)

        assert isinstance(cfg.device, str)
        assert isinstance(cfg.cache_policy, str)
        assert isinstance(cfg.max_len, int)
        assert isinstance(cfg.batch_size, int)
        assert isinstance(cfg.window_ai_threshold, float)
        assert isinstance(cfg.prefer_bf16, bool)

        print(f"✓ repo_id: {cfg.repo_id}")
        print(f"✓ device: {cfg.device}")
        print(f"✓ cache_policy: {cfg.cache_policy}")
        print(f"✓ max_len: {cfg.max_len}")
        print(f"✓ batch_size: {cfg.batch_size}")
        print(f"✓ window_ai_threshold: {cfg.window_ai_threshold}")
        print(f"✓ prefer_bf16: {cfg.prefer_bf16}")

        return True
    except Exception as e:
        print(f"✗ Default config value check failed: {e}")
        return False


def test_model_info_before_load():
    """Test model info before loading models"""
    print("\nTesting model info before load...")

    try:
        classifier = TextClassifier()

        loaded = classifier.are_models_loaded()
        info = classifier.get_model_info()

        assert loaded is False
        assert isinstance(info, dict)
        assert "status" in info
        assert "models_loaded" in info
        assert "repo_id" in info
        assert "device" in info
        assert "batch_size" in info

        print(f"✓ are_models_loaded(): {loaded}")
        print(f"✓ get_model_info()['status']: {info['status']}")
        print(f"✓ get_model_info()['models_loaded']: {info['models_loaded']}")

        return True
    except Exception as e:
        print(f"✗ Model info check failed: {e}")
        return False


def test_classify_requires_load():
    """Test that classify_texts_batch fails cleanly if models are not loaded"""
    print("\nTesting classification guardrail...")

    try:
        classifier = TextClassifier()

        try:
            classifier.classify_texts_batch(["hello world"])
            print("✗ classify_texts_batch should have failed before load_models()")
            return False
        except RuntimeError as e:
            print(f"✓ classify_texts_batch correctly guards before load: {e}")
            return True
        except Exception as e:
            print(f"✗ classify_texts_batch raised unexpected exception type: {e}")
            return False

    except Exception as e:
        print(f"✗ Classification guardrail test failed: {e}")
        return False


def test_input_coercion_helper():
    """Test text coercion helper behavior"""
    print("\nTesting input coercion helper...")

    try:
        classifier = TextClassifier()

        cases = [
            (None, ""),
            (" hello ", "hello"),
            (123, "123"),
            ("", ""),
        ]

        for raw, expected in cases:
            got = classifier._coerce_text(raw)
            assert got == expected, f"Expected {expected!r}, got {got!r}"

        print("✓ _coerce_text handles None / strings / non-strings correctly")
        return True
    except Exception as e:
        print(f"✗ Input coercion helper test failed: {e}")
        return False


def test_output_shape_contract_without_model_load():
    """
    Test expected output field contract by monkeypatching the internal detect function.
    This avoids requiring real model downloads/HF access.
    """
    print("\nTesting output shape contract (mocked)...")

    try:
        classifier = TextClassifier()

        def mock_detect_batch(
            texts,
            prediction_ids = None,
            lang = None,
            repo_id = None,
            subdir_by_lang = None,
            revision = None,
            hf_token = None,
            max_len = None,
            batch_size = None,
            progress = None,
            return_text = None,
            device = None,
            cache_policy = None,
            window_ai_threshold = None,
            prefer_bf16 = None,
        ):
            rows = []
            for i, _ in enumerate(texts):
                rows.append({
                    "prediction_id": prediction_ids[i] if prediction_ids else str(i),
                    "lang": lang[i] if isinstance(lang, list) else (lang or "en"),
                    "prediction_short": "AI",
                    "prediction_long": "Likely AI-generated",
                    "fraction_ai": 0.8,
                    "n_windows": 4,
                    "n_ai_segments": 3,
                    "n_human_segments": 1,
                    "n_tokens": 128,
                })
            return rows

        classifier._detect_batch = mock_detect_batch
        classifier._loaded = True

        texts = ["sample text one", "sample text two"]
        langs = ["en", "es"]
        prediction_ids = ["0", "1"]

        out = classifier.classify_texts_batch(
            texts,
            langs = langs,
            prediction_ids = prediction_ids,
        )

        assert isinstance(out, list)
        assert len(out) == 2

        required_keys = {
            "status",
            "prediction_id",
            "lang",
            "prediction_short",
            "prediction_long",
            "fraction_ai",
            "ai_probability",
            "human_probability",
            "n_windows",
            "n_ai_segments",
            "n_human_segments",
            "n_tokens",
        }

        for row in out:
            assert isinstance(row, dict)
            missing = required_keys.difference(row.keys())
            assert not missing, f"Missing keys: {missing}"

        print(f"✓ Returned {len(out)} rows")
        print(f"✓ Output keys: {sorted(out[0].keys())}")
        print(f"✓ Sample row: {out[0]}")

        return True
    except Exception as e:
        print(f"✗ Output shape contract test failed: {e}")
        return False


if __name__ == "__main__":
    print("Text Classification Core Test Suite")
    print("=" * 50)

    success = True
    success &= test_core_structure()
    success &= test_default_config_values()
    success &= test_model_info_before_load()
    success &= test_classify_requires_load()
    success &= test_input_coercion_helper()
    success &= test_output_shape_contract_without_model_load()

    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! Core is ready.")
    else:
        print("✗ Some tests failed. Check the issues above.")