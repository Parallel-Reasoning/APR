import sys
from pathlib import Path

import pytest


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""
    
    def test_pytest_installed(self):
        """Test that pytest is installed and importable."""
        import pytest
        assert pytest.__version__
    
    def test_pytest_cov_installed(self):
        """Test that pytest-cov is installed and importable."""
        import pytest_cov
        assert pytest_cov
    
    def test_pytest_mock_installed(self):
        """Test that pytest-mock is installed and importable."""
        import pytest_mock
        assert pytest_mock
    
    def test_src_module_importable(self):
        """Test that the src module is importable."""
        import src
        assert src
    
    def test_fixtures_available(self, temp_dir, mock_config, mock_logger):
        """Test that custom fixtures are available and working."""
        assert temp_dir.exists()
        assert isinstance(mock_config, dict)
        assert hasattr(mock_logger, 'info')
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker is recognized."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker is recognized."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker is recognized."""
        assert True
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture creates a directory."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
    
    def test_mock_environment_fixture(self, mock_environment):
        """Test that mock_environment fixture sets environment variables."""
        import os
        assert os.environ.get("TEST_API_KEY") == "test-key-123"
        assert os.environ.get("TEST_MODEL") == "test-model"
        assert os.environ.get("TEST_ENV") == "testing"
    
    def test_sample_data_fixture(self, sample_data):
        """Test that sample_data fixture provides expected structure."""
        assert "inputs" in sample_data
        assert "outputs" in sample_data
        assert "metadata" in sample_data
        assert len(sample_data["inputs"]) == 3
    
    def test_mock_model_fixture(self, mock_model):
        """Test that mock_model fixture provides expected methods."""
        result = mock_model.predict("test")
        assert mock_model.predict.called
        
        eval_result = mock_model.evaluate()
        assert "accuracy" in eval_result
        assert "loss" in eval_result
    
    def test_mock_file_system_fixture(self, mock_file_system):
        """Test that mock_file_system fixture creates expected structure."""
        assert mock_file_system["data_dir"].exists()
        assert mock_file_system["output_dir"].exists()
        assert mock_file_system["config_dir"].exists()
        assert mock_file_system["logs_dir"].exists()
        
        config_file = mock_file_system["config_dir"] / "config.yaml"
        assert config_file.exists()
        assert "test: true" in config_file.read_text()


def test_basic_assertion():
    """Basic test to ensure pytest runs."""
    assert 2 + 2 == 4


def test_python_version():
    """Test Python version meets requirements."""
    assert sys.version_info >= (3, 10)