"""Tests for the Basilica GPU compute tools and local training guard."""

import json
import os
import tempfile

import pytest

from ralph_loop.basilica import (
    ALLOWED_GPU_TYPES,
    BASILICA_TOOLS,
    check_for_local_training,
    execute_tool_call,
)


class TestLocalTrainingGuard:
    """Test that local GPU training is blocked."""

    def test_blocks_python_train(self):
        assert check_for_local_training("python train.py") is not None

    def test_blocks_python_training_script(self):
        assert check_for_local_training("python training/train_model.py --epochs 10") is not None

    def test_blocks_search_py(self):
        assert check_for_local_training("python search.py --config grid") is not None

    def test_blocks_backward(self):
        assert check_for_local_training("loss.backward()") is not None

    def test_blocks_optimizer_step(self):
        assert check_for_local_training("optimizer.step()") is not None

    def test_allows_pip_install_torch(self):
        assert check_for_local_training("pip install torch pandas numpy") is None

    def test_allows_torch_load(self):
        assert check_for_local_training("torch.load('model.pt')") is None

    def test_allows_data_pipeline(self):
        assert check_for_local_training("python data_pipeline.py") is None

    def test_allows_evaluation(self):
        assert check_for_local_training("python evaluate.py --model best") is None

    def test_allows_validator_emulator(self):
        assert check_for_local_training("python validator_emulator.py") is None

    def test_allows_leaderboard(self):
        assert check_for_local_training("python leaderboard.py --show") is None

    def test_allows_curl(self):
        assert check_for_local_training("curl https://api.synth.com/scores") is None

    def test_allows_mkdir(self):
        assert check_for_local_training("mkdir -p models data") is None


class TestToolDefinitions:
    """Test that tool definitions are well-formed."""

    def test_tools_exist(self):
        assert len(BASILICA_TOOLS) == 4

    def test_tool_names(self):
        names = {t["function"]["name"] for t in BASILICA_TOOLS}
        assert names == {
            "basilica_submit_job",
            "basilica_check_job",
            "basilica_fetch_results",
            "basilica_list_jobs",
        }

    def test_submit_job_has_required_params(self):
        submit_tool = next(
            t for t in BASILICA_TOOLS if t["function"]["name"] == "basilica_submit_job"
        )
        required = submit_tool["function"]["parameters"]["required"]
        assert "script_content" in required
        assert "gpu_type" in required
        assert "job_name" in required

    def test_gpu_types_restricted(self):
        submit_tool = next(
            t for t in BASILICA_TOOLS if t["function"]["name"] == "basilica_submit_job"
        )
        gpu_enum = submit_tool["function"]["parameters"]["properties"]["gpu_type"]["enum"]
        assert set(gpu_enum) == ALLOWED_GPU_TYPES


class TestToolExecution:
    """Test tool call execution."""

    def test_submit_job_creates_files(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = execute_tool_call(
                "basilica_submit_job",
                {
                    "script_content": "import torch\nprint('training')",
                    "gpu_type": "A4000",
                    "job_name": "test_job",
                },
                workspace,
            )
            data = json.loads(result)
            assert "job_id" in data
            assert data["gpu_type"] == "A4000"

            # Check files were created
            jobs_dir = os.path.join(workspace, "basilica_jobs")
            assert os.path.isdir(jobs_dir)

            # Find the job dir
            job_dirs = os.listdir(jobs_dir)
            assert len(job_dirs) == 1

            job_dir = os.path.join(jobs_dir, job_dirs[0])
            assert os.path.exists(os.path.join(job_dir, "train.py"))
            assert os.path.exists(os.path.join(job_dir, "job_meta.json"))

    def test_submit_job_rejects_expensive_gpu(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = execute_tool_call(
                "basilica_submit_job",
                {
                    "script_content": "print('hi')",
                    "gpu_type": "A100",
                    "job_name": "expensive_job",
                },
                workspace,
            )
            data = json.loads(result)
            assert "error" in data
            assert "A100" in data["error"]

    def test_submit_job_saves_requirements(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = execute_tool_call(
                "basilica_submit_job",
                {
                    "script_content": "print('train')",
                    "gpu_type": "V100",
                    "job_name": "req_test",
                    "requirements": "torch\npandas\nnumpy",
                },
                workspace,
            )
            data = json.loads(result)
            job_id = data["job_id"]

            req_path = os.path.join(
                workspace, "basilica_jobs", job_id, "requirements.txt"
            )
            assert os.path.exists(req_path)
            with open(req_path) as f:
                assert "torch" in f.read()

    def test_check_nonexistent_job(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = execute_tool_call(
                "basilica_check_job",
                {"job_id": "nonexistent_123"},
                workspace,
            )
            data = json.loads(result)
            assert "error" in data

    def test_list_jobs_empty(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = execute_tool_call(
                "basilica_list_jobs",
                {},
                workspace,
            )
            data = json.loads(result)
            assert "jobs" in data

    def test_unknown_tool(self):
        with tempfile.TemporaryDirectory() as workspace:
            result = execute_tool_call(
                "nonexistent_tool",
                {},
                workspace,
            )
            data = json.loads(result)
            assert "error" in data


class TestLLMResponse:
    """Test the LLMResponse dataclass."""

    def test_import(self):
        from ralph_loop.llm import LLMResponse

        r = LLMResponse(content="hello")
        assert r.content == "hello"
        assert r.has_tool_calls is False

    def test_with_tool_calls(self):
        from ralph_loop.llm import LLMResponse

        r = LLMResponse(
            content="submitting job",
            tool_calls=[{"id": "1", "name": "basilica_submit_job", "arguments": {}}],
        )
        assert r.has_tool_calls is True
