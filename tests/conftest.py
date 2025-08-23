import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is automatically cleaned up."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Provide a temporary file that is automatically cleaned up."""
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("test content")
    yield temp_file


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary."""
    return {
        "model": "test-model",
        "temperature": 0.7,
        "max_tokens": 100,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
        "seed": 42,
        "debug": False,
    }


@pytest.fixture
def mock_environment(monkeypatch) -> Dict[str, str]:
    """Mock environment variables for testing."""
    env_vars = {
        "TEST_API_KEY": "test-key-123",
        "TEST_MODEL": "test-model",
        "TEST_ENV": "testing",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def mock_logger() -> MagicMock:
    """Provide a mock logger for testing logging behavior."""
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    return logger


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Provide sample data for testing."""
    return {
        "inputs": ["test input 1", "test input 2", "test input 3"],
        "outputs": ["output 1", "output 2", "output 3"],
        "metadata": {
            "version": "1.0.0",
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "test",
        },
    }


@pytest.fixture
def mock_model() -> MagicMock:
    """Provide a mock model for testing."""
    model = MagicMock()
    model.predict = MagicMock(return_value={"prediction": "test", "confidence": 0.95})
    model.train = MagicMock()
    model.evaluate = MagicMock(return_value={"accuracy": 0.85, "loss": 0.15})
    model.save = MagicMock()
    model.load = MagicMock()
    return model


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    yield


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    
    captured_output = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured_output)
    return captured_output


@pytest.fixture
def mock_file_system(temp_dir: Path) -> Dict[str, Path]:
    """Create a mock file system structure for testing."""
    structure = {
        "data_dir": temp_dir / "data",
        "output_dir": temp_dir / "output",
        "config_dir": temp_dir / "config",
        "logs_dir": temp_dir / "logs",
    }
    
    for dir_path in structure.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    (structure["config_dir"] / "config.yaml").write_text("test: true\n")
    (structure["data_dir"] / "sample.txt").write_text("sample data\n")
    
    return structure