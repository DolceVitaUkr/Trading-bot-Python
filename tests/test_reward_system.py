import pytest
from modules.reward_system import RewardSystem

@pytest.fixture
def reward_system():
    """Fixture for a RewardSystem instance."""
    return RewardSystem(starting_balance=1000.0, points_multiplier=2.0)

class TestRewardSystem:
    def test_initialization(self):
        """Tests that the RewardSystem initializes correctly."""
        rs = RewardSystem(starting_balance=500, points_multiplier=1.5)
        assert rs.wallet_balance == 500
        assert rs.total_points == 0
        assert rs.points_multiplier == 1.5

    def test_update_with_profit(self, reward_system):
        """Tests the update method with a profitable trade."""
        initial_balance = reward_system.wallet_balance
        initial_points = reward_system.total_points

        pnl_net_usd = 50.0
        context = {"asset": "SPOT", "mode": "TREND"}

        reward_system.update(pnl_net_usd, context)

        assert reward_system.wallet_balance == initial_balance + 50.0
        # Points = 50.0 * 2.0 = 100.0
        assert reward_system.total_points == initial_points + 100.0

    def test_update_with_loss(self, reward_system):
        """Tests the update method with a losing trade."""
        initial_balance = reward_system.wallet_balance
        initial_points = reward_system.total_points

        pnl_net_usd = -25.0
        context = {"asset": "PERP", "mode": "SCALP"}

        reward_system.update(pnl_net_usd, context)

        assert reward_system.wallet_balance == initial_balance - 25.0
        # Points = -25.0 * 2.0 = -50.0
        assert reward_system.total_points == initial_points - 50.0

    def test_multiple_updates(self, reward_system):
        """Tests a sequence of updates."""
        reward_system.update(100, {}) # bal=1100, pts=200
        reward_system.update(-30, {}) # bal=1070, pts=140
        reward_system.update(5, {})   # bal=1075, pts=150

        assert reward_system.wallet_balance == 1075.0
        assert reward_system.total_points == 150.0

    def test_get_snapshot(self, reward_system):
        """Tests the snapshot functionality."""
        reward_system.update(20, {})
        snapshot = reward_system.get_snapshot()

        assert snapshot["wallet_balance"] == 1020.0
        assert snapshot["total_points"] == 40.0
