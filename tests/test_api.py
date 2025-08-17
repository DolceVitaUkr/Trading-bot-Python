import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# The app from your API
from api.main import app, get_branch_manager
from core.schemas import BranchStatus

# Create a TestClient instance
client = TestClient(app)

@pytest.fixture
def mock_branch_manager():
    """Pytest fixture to create a mock BranchManager."""
    manager = MagicMock()

    # Mock branches
    branch_spot = MagicMock()
    branch_spot.product_name = "CRYPTO_SPOT"
    branch_spot.status = BranchStatus.RUNNING

    branch_futures = MagicMock()
    branch_futures.product_name = "CRYPTO_FUTURES"
    branch_futures.status = BranchStatus.STOPPED

    # Mock manager methods
    manager.get_all_statuses.return_value = {
        "CRYPTO_SPOT": "running",
        "CRYPTO_FUTURES": "stopped"
    }
    manager.get_branch.side_effect = lambda name: {
        "CRYPTO_SPOT": branch_spot,
        "CRYPTO_FUTURES": branch_futures
    }.get(name)

    return manager

def test_get_all_statuses(mock_branch_manager):
    """Test the endpoint for getting all branch statuses."""
    # Override the dependency with our mock
    app.dependency_overrides[get_branch_manager] = lambda: mock_branch_manager

    response = client.get("/branches/status")
    assert response.status_code == 200
    assert response.json() == {
        "CRYPTO_SPOT": "running",
        "CRYPTO_FUTURES": "stopped"
    }

    # Clean up the override
    app.dependency_overrides = {}

def test_get_specific_branch_status_success(mock_branch_manager):
    """Test getting the status of a specific, existing branch."""
    app.dependency_overrides[get_branch_manager] = lambda: mock_branch_manager

    response = client.get("/branches/CRYPTO_SPOT/status")
    assert response.status_code == 200
    assert response.json() == {"name": "CRYPTO_SPOT", "status": "running"}

    app.dependency_overrides = {}

def test_get_specific_branch_status_not_found(mock_branch_manager):
    """Test getting the status of a non-existent branch."""
    app.dependency_overrides[get_branch_manager] = lambda: mock_branch_manager

    response = client.get("/branches/NON_EXISTENT/status")
    assert response.status_code == 404
    assert response.json() == {"detail": "Branch 'NON_EXISTENT' not found."}

    app.dependency_overrides = {}

def test_root_endpoint():
    """Test the root health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Trading Bot API is running."}
