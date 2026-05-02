import os
import pytest
from panoramix_core.demo_mode import is_force_cache, set_force_cache_for_test


def setup_function():
    set_force_cache_for_test(None)
    os.environ.pop("DEMO_FORCE_CACHE", None)


def test_default_off():
    assert is_force_cache() is False


def test_env_var_on():
    os.environ["DEMO_FORCE_CACHE"] = "1"
    assert is_force_cache() is True


def test_env_var_off_string():
    os.environ["DEMO_FORCE_CACHE"] = "0"
    assert is_force_cache() is False


def test_set_for_test_overrides_env():
    set_force_cache_for_test(True)
    assert is_force_cache() is True
    set_force_cache_for_test(False)
    assert is_force_cache() is False
