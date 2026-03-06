"""Tests for the architecture comparison experiment script."""

import subprocess
import sys


class TestCompareArchitectures:
    """Test the compare_architectures example script."""

    def test_script_runs_successfully(self) -> None:
        """Script completes without errors on minimal config."""
        result = subprocess.run(
            [
                sys.executable,
                'examples/compare_architectures.py',
                '--epochs',
                '1',
                '--num-samples',
                '32',
                '--batch-size',
                '8',
                '--embed-dim',
                '32',
                '--num-layers',
                '1',
                '--seq-len',
                '16',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f'Script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}'
        )
        assert 'SUMMARY' in result.stdout
        assert 'GPT' in result.stdout
        assert 'LLaMA' in result.stdout
        assert 'Mixtral' in result.stdout
        assert 'DeepSeek-V2' in result.stdout
        assert 'Mamba' in result.stdout

    def test_script_prints_param_counts(self) -> None:
        """Script prints parameter counts for all models."""
        result = subprocess.run(
            [
                sys.executable,
                'examples/compare_architectures.py',
                '--epochs',
                '1',
                '--num-samples',
                '16',
                '--batch-size',
                '8',
                '--embed-dim',
                '32',
                '--num-layers',
                '1',
                '--seq-len',
                '16',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert 'Model Parameter Counts' in result.stdout
        assert 'params' in result.stdout
