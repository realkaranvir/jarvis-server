from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from jarvis_server.dependencies import ConfigDep, HttpClientDep
from jarvis_server.schemas import STTResponse
from jarvis_server.services.providers import ProviderError, transcribe


router = APIRouter(prefix="/stt", tags=["stt"])


@router.post("")
async def transcribe_audio(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    config: ConfigDep,
    client: HttpClientDep,
) -> STTResponse:
    server = config.stt
    contents = await audio.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty",
        )

    try:
        text = await transcribe(
            contents,
            audio.filename or "audio.wav",
            audio.content_type or "application/octet-stream",
            server,
            client,
        )
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="STT provider request failed",
        ) from exc

    return STTResponse(text=text, model=server.model)
