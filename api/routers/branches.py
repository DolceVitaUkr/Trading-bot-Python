from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any
from pydantic import BaseModel

from managers.branch_manager import BranchManager
from api.main import get_branch_manager

router = APIRouter(
    prefix="/branches",
    tags=["branches"],
)

class ModeRequest(BaseModel):
    mode: str

@router.get("/status", response_model=Dict[str, str])
async def get_all_branch_statuses(
    branch_manager: BranchManager = Depends(get_branch_manager)
):
    """Get the running status of all managed branches."""
    return branch_manager.get_all_statuses()

@router.get("/{branch_name}/status", response_model=Dict[str, str])
async def get_branch_status(
    branch_name: str,
    branch_manager: BranchManager = Depends(get_branch_manager)
):
    """Get the running status of a specific branch."""
    branch = branch_manager.get_branch(branch_name)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch '{branch_name}' not found.")
    return {"name": branch.product_name, "status": branch.status.value}

@router.get("/{branch_name}/balances", response_model=Dict[str, Any])
async def get_branch_balances(
    branch_name: str,
    branch_manager: BranchManager = Depends(get_branch_manager)
):
    """Get the wallet balances for a specific branch."""
    branch = branch_manager.get_branch(branch_name)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch '{branch_name}' not found.")

    fetcher = branch.pipeline.get("account_fetcher")
    if not hasattr(fetcher, 'get_wallet_balance'):
        raise HTTPException(status_code=501, detail="Balance fetching not implemented for this branch.")

    # Determine account type based on product
    account_type = "UNIFIED" if "FUTURES" in branch_name else "SPOT"
    if branch.broker == "BYBIT":
        account_type = "UNIFIED"

    balances = fetcher.get_wallet_balance(account_type=account_type)
    return balances

@router.get("/{branch_name}/positions", response_model=Dict[str, Any])
async def get_branch_positions(
    branch_name: str,
    branch_manager: BranchManager = Depends(get_branch_manager)
):
    """Get the open positions for a specific branch."""
    branch = branch_manager.get_branch(branch_name)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch '{branch_name}' not found.")

    fetcher = branch.pipeline.get("account_fetcher")
    if not hasattr(fetcher, 'get_positions'):
        raise HTTPException(status_code=501, detail="Position fetching not implemented for this branch.")

    category = "linear" if "FUTURES" in branch_name else "spot"
    positions = fetcher.get_positions(category=category)
    return positions

@router.post("/{branch_name}/start")
async def start_branch(
    branch_name: str,
    branch_manager: BranchManager = Depends(get_branch_manager)
):
    """Start a specific branch."""
    branch = branch_manager.get_branch(branch_name)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch '{branch_name}' not found.")
    branch.start()
    return {"message": f"Branch '{branch_name}' started."}

@router.post("/{branch_name}/stop")
async def stop_branch(
    branch_name: str,
    branch_manager: BranchManager = Depends(get_branch_manager)
):
    """Stop a specific branch."""
    branch = branch_manager.get_branch(branch_name)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch '{branch_name}' not found.")
    branch.stop()
    return {"message": f"Branch '{branch_name}' stopped."}

@router.post("/{branch_name}/restart")
async def restart_branch(
    branch_name: str,
    branch_manager: BranchManager = Depends(get_branch_manager)
):
    """Restart a specific branch."""
    branch = branch_manager.get_branch(branch_name)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch '{branch_name}' not found.")
    branch.stop()
    # In a real-world scenario, you might want a delay or to check if it stopped cleanly
    branch.start()
    return {"message": f"Branch '{branch_name}' restarted."}

@router.post("/{branch_name}/mode")
async def set_branch_mode(
    branch_name: str,
    mode_request: ModeRequest,
    branch_manager: BranchManager = Depends(get_branch_manager)
):
    """Set the mode (paper/live) for a specific branch."""
    branch = branch_manager.get_branch(branch_name)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch '{branch_name}' not found.")

    branch.set_mode(mode_request.mode)
    return {"message": f"Mode for branch '{branch_name}' set to {mode_request.mode}."}

@router.post("/{branch_name}/kill_switch")
async def trigger_kill_switch(
    branch_name: str,
    branch_manager: BranchManager = Depends(get_branch_manager)
):
    """Trigger the kill switch for a specific branch."""
    branch = branch_manager.get_branch(branch_name)
    if not branch:
        raise HTTPException(status_code=404, detail=f"Branch '{branch_name}' not found.")

    branch.trigger_kill_switch()
    return {"message": f"Kill switch for branch '{branch_name}' triggered."}
