from fastapi import APIRouter

router = APIRouter(
    prefix="/telemetry",
    tags=["telemetry"],
)

@router.get("/")
async def get_telemetry():
    """
    Returns telemetry data. (STUB)
    """
    return {"status": "This is a stub endpoint for telemetry. Implementation is pending."}
