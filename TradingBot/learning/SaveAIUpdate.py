import logging
import pickle
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

BASE_MODEL_DIR = Path("models")

def save_model_artifact(
    product_name: str,
    strategy_id: str,
    artifact: Any
) -> Path:
    """
    Saves a trained model artifact to a product-specific directory.

    Args:
        product_name: The product, e.g., "forex_spot", "forex_options".
        strategy_id: A unique identifier for the strategy.
        artifact: The Python object to save (e.g., model, dict of parameters).

    Returns:
        The path to the saved file.
    """
    product_name = product_name.lower()
    product_dir = BASE_MODEL_DIR / product_name
    product_dir.mkdir(parents=True, exist_ok=True)

    file_path = product_dir / f"{strategy_id}.pkl"

    log.info(f"Saving model artifact for product '{product_name}' to {file_path}")

    try:
        with open(file_path, "wb") as f:
            pickle.dump(artifact, f)
        log.info("Artifact saved successfully.")
        return file_path
    except (pickle.PicklingError, IOError) as e:
        log.error(f"Failed to save model artifact to {file_path}: {e}")
        raise

def load_model_artifact(
    product_name: str,
    strategy_id: str
) -> Any:
    """
    Loads a model artifact from a product-specific directory.
    """
    product_name = product_name.lower()
    file_path = BASE_MODEL_DIR / product_name / f"{strategy_id}.pkl"

    if not file_path.exists():
        log.warning(f"Model artifact not found at {file_path}")
        return None

    log.info(f"Loading model artifact from {file_path}")
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, IOError) as e:
        log.error(f"Failed to load model artifact from {file_path}: {e}")
        raise
