from fastapi import APIRouter


router = APIRouter(tags=["system"])


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "Jarvis server is running"}
