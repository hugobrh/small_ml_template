import os
import pytest
from pathlib import Path
import joblib
from src.credit_model.train import train

def test_train_creates_artifacts(tmp_path, monkeypatch):
    """
    Tests if the train function runs successfully and saves the model.
    Using monkeypatch to ensure files are saved in a temporary directory.
    """
    # Change the current working directory to a temporary path for the test
    monkeypatch.chdir(tmp_path)
    
    # Run the training function
    # Note: This will download the dataset if not cached, which might take a moment
    train()

    # 1. Check if the 'models' directory was created
    model_dir = tmp_path / "models"
    assert model_dir.is_dir()

    # 2. Check if the model file exists
    model_file = model_dir / "lr.joblib"
    assert model_file.exists()

    # 3. Load the model and verify it's a valid scikit-learn pipeline
    loaded_model = joblib.load(model_file)
    assert hasattr(loaded_model, "predict"), "Loaded model should have a predict method"
    assert "preprocessor" in loaded_model.named_steps
    assert "classifier" in loaded_model.named_steps

def test_model_accuracy_threshold(tmp_path, monkeypatch, capsys):
    """
    Tests if the model achieves a minimum accuracy threshold.
    We parse the printed output from the train function.
    """
    monkeypatch.chdir(tmp_path)
    
    train()
    
    # Capture the printed output
    captured = capsys.readouterr()
    
    # Extract accuracy from the print statement: "LR model trained with accuracy: 0.XXX"
    output = captured.out
    assert "LR model trained with accuracy" in output
    
    accuracy_str = output.split(":")[-1].strip()
    accuracy = float(accuracy_str)
    
    # Assert a reasonable baseline for the German Credit dataset
    assert accuracy > 0.6, f"Model accuracy {accuracy} is lower than expected threshold."