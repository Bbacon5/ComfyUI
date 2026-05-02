"""Tests that workflow_id is included alongside prompt_id in WebSocket payloads
emitted by the progress handler and the prompt executor.

Frontend stores extra_data["extra_pnginfo"]["workflow"]["id"] when queueing a
prompt; we propagate that as `workflow_id` on every execution event so a
multi-tab UI can scope progress state by workflow even when terminal
WebSocket frames are dropped.
"""

from unittest.mock import MagicMock

import pytest

from comfy_execution.progress import (
    NodeState,
    ProgressRegistry,
    WebUIProgressHandler,
    reset_progress_state,
    get_progress_state,
)


class _DummyDynPrompt:
    def get_display_node_id(self, node_id):
        return node_id

    def get_parent_node_id(self, node_id):
        return None

    def get_real_node_id(self, node_id):
        return node_id


@pytest.fixture
def server():
    s = MagicMock()
    s.client_id = "client-1"
    return s


def _registry(workflow_id):
    return ProgressRegistry(
        prompt_id="prompt-1",
        dynprompt=_DummyDynPrompt(),
        workflow_id=workflow_id,
    )


class TestProgressStatePayload:
    def test_progress_state_includes_workflow_id(self, server):
        registry = _registry("wf-abc")
        registry.nodes["n1"] = {
            "state": NodeState.Running,
            "value": 1.0,
            "max": 5.0,
        }

        handler = WebUIProgressHandler(server)
        handler.set_registry(registry)
        handler._send_progress_state("prompt-1", registry.nodes)

        server.send_sync.assert_called_once()
        event, payload, sid = server.send_sync.call_args.args
        assert event == "progress_state"
        assert payload["prompt_id"] == "prompt-1"
        assert payload["workflow_id"] == "wf-abc"
        assert payload["nodes"]["n1"]["workflow_id"] == "wf-abc"
        assert payload["nodes"]["n1"]["prompt_id"] == "prompt-1"
        assert sid == "client-1"

    def test_progress_state_workflow_id_none_when_missing(self, server):
        registry = _registry(None)
        registry.nodes["n1"] = {
            "state": NodeState.Running,
            "value": 0.5,
            "max": 1.0,
        }

        handler = WebUIProgressHandler(server)
        handler.set_registry(registry)
        handler._send_progress_state("prompt-1", registry.nodes)

        _, payload, _ = server.send_sync.call_args.args
        assert payload["workflow_id"] is None
        assert payload["nodes"]["n1"]["workflow_id"] is None


class TestProgressRegistryConstruction:
    def test_workflow_id_default_is_none(self):
        registry = ProgressRegistry(
            prompt_id="prompt-1", dynprompt=_DummyDynPrompt()
        )
        assert registry.workflow_id is None

    def test_workflow_id_stored_on_registry(self):
        registry = ProgressRegistry(
            prompt_id="prompt-1",
            dynprompt=_DummyDynPrompt(),
            workflow_id="wf-xyz",
        )
        assert registry.workflow_id == "wf-xyz"


class TestResetProgressState:
    def test_reset_threads_workflow_id(self):
        reset_progress_state("prompt-1", _DummyDynPrompt(), "wf-456")
        assert get_progress_state().workflow_id == "wf-456"

    def test_reset_default_workflow_id_none(self):
        reset_progress_state("prompt-2", _DummyDynPrompt())
        assert get_progress_state().workflow_id is None


class TestExecutionMessagePayloadsContainWorkflowId:
    """Static-analysis guard ensuring every WebSocket message payload that
    carries `prompt_id` also carries `workflow_id`. This is a regression net
    for future refactors of execution.py / main.py / progress.py and avoids
    the GPU/torch dependency of importing `execution.py` directly.
    """

    @staticmethod
    def _emitting_dicts(source: str):
        """Yield every dict literal in `source` that contains a 'prompt_id' key."""
        import ast

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                k.value
                for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
            if "prompt_id" in keys:
                yield node, keys

    def _assert_workflow_id_in_every_prompt_id_dict(self, file_path: str):
        from pathlib import Path

        source = Path(file_path).read_text()
        offenders = []
        for node, keys in self._emitting_dicts(source):
            if "workflow_id" not in keys:
                offenders.append((node.lineno, keys))
        assert not offenders, (
            f"{file_path}: dict literals with 'prompt_id' but no 'workflow_id': {offenders}"
        )

    def test_execution_py_payloads_include_workflow_id(self):
        self._assert_workflow_id_in_every_prompt_id_dict("execution.py")

    def test_main_py_payloads_include_workflow_id(self):
        self._assert_workflow_id_in_every_prompt_id_dict("main.py")

    def test_progress_py_payloads_include_workflow_id(self):
        self._assert_workflow_id_in_every_prompt_id_dict("comfy_execution/progress.py")


class TestPreviewImageMetadataPayload:
    """Verify PREVIEW_IMAGE_WITH_METADATA metadata carries workflow_id."""

    def test_preview_metadata_includes_workflow_id(self):
        from unittest.mock import MagicMock, patch
        from PIL import Image

        from comfy_execution.progress import (
            NodeState,
            ProgressRegistry,
            WebUIProgressHandler,
        )

        class _DynPrompt:
            def get_display_node_id(self, n):
                return n

            def get_parent_node_id(self, n):
                return None

            def get_real_node_id(self, n):
                return n

        server = MagicMock()
        server.client_id = "cid"
        server.sockets_metadata = {}

        registry = ProgressRegistry(
            prompt_id="p1", dynprompt=_DynPrompt(), workflow_id="wf-1"
        )
        handler = WebUIProgressHandler(server)
        handler.set_registry(registry)

        image = ("PNG", Image.new("RGB", (1, 1)), None)

        with patch(
            "comfy_execution.progress.feature_flags.supports_feature",
            return_value=True,
        ):
            handler.update_handler(
                node_id="n1",
                value=1.0,
                max_value=1.0,
                state={
                    "state": NodeState.Running,
                    "value": 1.0,
                    "max": 1.0,
                },
                prompt_id="p1",
                image=image,
            )

        preview_calls = [
            c
            for c in server.send_sync.call_args_list
            if c.args[0] != "progress_state"
        ]
        assert len(preview_calls) == 1
        _, payload, _ = preview_calls[0].args
        _, metadata = payload
        assert metadata["prompt_id"] == "p1"
        assert metadata["workflow_id"] == "wf-1"
