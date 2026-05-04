"""Tests for comfy.deploy_environment."""

import os

import pytest

from comfy.deploy_environment import get_deploy_environment


@pytest.fixture(autouse=True)
def _reset_cache_and_base_path(tmp_path, monkeypatch):
    """Reset the functools cache and point folder_paths.base_path at a tmp dir for each test."""
    get_deploy_environment.cache_clear()
    import folder_paths
    monkeypatch.setattr(folder_paths, "base_path", str(tmp_path))
    yield
    get_deploy_environment.cache_clear()


def _write_env_file(tmp_path, content: str) -> str:
    path = os.path.join(str(tmp_path), ".comfy_environment")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class TestGetDeployEnvironment:
    def test_returns_local_git_when_file_missing(self):
        assert get_deploy_environment() == "local_git"

    def test_reads_value_from_file(self, tmp_path):
        _write_env_file(tmp_path, "local_desktop2_standalone\n")
        assert get_deploy_environment() == "local_desktop2_standalone"

    def test_strips_trailing_whitespace_and_newline(self, tmp_path):
        _write_env_file(tmp_path, "  local_desktop2_standalone  \n")
        assert get_deploy_environment() == "local_desktop2_standalone"

    def test_only_first_line_is_used(self, tmp_path):
        _write_env_file(tmp_path, "first_line\nsecond_line\n")
        assert get_deploy_environment() == "first_line"

    def test_empty_file_falls_back_to_default(self, tmp_path):
        _write_env_file(tmp_path, "")
        assert get_deploy_environment() == "local_git"

    def test_empty_after_whitespace_strip_falls_back_to_default(self, tmp_path):
        _write_env_file(tmp_path, "   \n")
        assert get_deploy_environment() == "local_git"

    def test_strips_control_chars_within_first_line(self, tmp_path):
        # Embedded NUL/control chars in the value should be stripped
        # (header-injection / smuggling protection).
        _write_env_file(tmp_path, "abc\x00\x07xyz\n")
        assert get_deploy_environment() == "abcxyz"

    def test_strips_non_ascii_characters(self, tmp_path):
        _write_env_file(tmp_path, "café-é\n")
        assert get_deploy_environment() == "caf-"

    def test_caps_read_at_128_bytes(self, tmp_path):
        # A single huge line with no newline must not be fully read into memory.
        huge = "x" * 10_000
        _write_env_file(tmp_path, huge)
        result = get_deploy_environment()
        assert result == "x" * 128

    def test_result_is_cached_across_calls(self, tmp_path):
        path = _write_env_file(tmp_path, "first_value\n")
        assert get_deploy_environment() == "first_value"
        # Overwrite the file — cached value should still be returned.
        with open(path, "w", encoding="utf-8") as f:
            f.write("second_value\n")
        assert get_deploy_environment() == "first_value"

    def test_unreadable_file_falls_back_to_default(self, tmp_path, monkeypatch):
        _write_env_file(tmp_path, "should_not_be_used\n")

        def _boom(*args, **kwargs):
            raise OSError("simulated read failure")

        monkeypatch.setattr("builtins.open", _boom)
        assert get_deploy_environment() == "local_git"
