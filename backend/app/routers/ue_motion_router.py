"""Relay body motion only to the local Unreal Editor's Unix socket."""

import os

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter(prefix="/api/ue-motion")
_MAX_BODY = 8 * 1024 * 1024
_SOCKET = os.getenv(
    "GAKUCHO_UE_MOTION_SOCKET", "/tmp/gakucho-ai-motion-ue/bridge/body.sock"
)


async def _relay(action: str, request: Request) -> Response:
    body = bytearray()
    if request.method == "POST":
        if (
            request.headers.get("content-type", "").split(";")[0].strip()
            != "application/json"
        ):
            raise HTTPException(415, "JSON が必要です")
        async for chunk in request.stream():
            if len(body) + len(chunk) > _MAX_BODY:
                raise HTTPException(413, "身体モーションが大きすぎます")
            body.extend(chunk)
    try:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=_SOCKET), timeout=15.0
        ) as client:
            reply = await client.request(
                request.method,
                f"http://editor/{action}",
                content=bytes(body),
                headers={"Content-Type": "application/json"},
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            504, "Unreal Editor の身体再生がタイムアウトしました"
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, "Unreal Editor の身体再生に接続できません") from exc
    return Response(
        reply.content, status_code=reply.status_code, media_type="application/json"
    )


@router.get("/health")
async def health(request: Request) -> Response:
    return await _relay("health", request)


@router.post("/play")
async def play(request: Request) -> Response:
    return await _relay("play", request)


@router.post("/stop")
async def stop(request: Request) -> Response:
    return await _relay("stop", request)
