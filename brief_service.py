import asyncio
import os
from datetime import datetime

import httpx

from gen_service import generate_brief, refine_brief
from test_preprocessing import text_preprocessing_pipeline

IMAGE_EXTRACTION_API = "https://sagdafathy--image-caption-app-predict.modal.run/"
AUDIO_EXTRACTION_API = "https://sagdafathy--raqify-whisper-api-fastapi-app.modal.run/transcribe"


async def handle_brief_request(request: dict) -> dict:
    if request["type"] == "INITIAL_BRIEF":
        return await handle_initial_brief(request)
    elif request["type"] == "REFINEMENT":
        return await handle_refinement(request)
    raise ValueError(f"Unknown request type: {request['type']}")


#  INITIAL BRIEF

async def handle_initial_brief(request: dict) -> dict:
    attachments = request.get("attachments", [])

    # Run attachment processing in parallel
    attachment_texts = await process_attachments(attachments)

   
    raw_text       = request.get("rawText", "")
    processed_text = text_preprocessing_pipeline(raw_text)

    result = generate_brief(processed_text, attachment_texts)

    return {
        "projectId": request["projectId"],
        "version": 1,
        **result,
        "metadata": {
            "attachmentsProcessed": len(attachments),
            "generatedAt": datetime.utcnow().isoformat(),
        },
    }



#  REFINEMENT

async def handle_refinement(request: dict) -> dict:
    latest  = request["latestVersion"]
    ref_req = request["refinementRequest"]

    new_attachments = await process_attachments(ref_req.get("attachments", []))

    processed_message = text_preprocessing_pipeline(ref_req.get("message", ""))

    result = refine_brief(
        current_brief = latest["formattedBrief"],
        current_version = latest["versionNumber"],
        version_history = request.get("versionHistory", []),
        timeline = request.get("timeline", []),
        raw_new_request = processed_message,
        new_attachments = new_attachments,
    )

    return {
        "briefId": request["briefId"],
        "version": latest["versionNumber"] + 1,
        **result,
        "metadata": {
            "refinedAt": datetime.utcnow().isoformat(),
            "previousVersion": latest["versionNumber"],
        },
    }


#  ATTACHMENT PROCESSING

async def process_attachments(attachments: list) -> list:
    if not attachments:
        return []
    tasks = [_process_one(att) for att in attachments]
    return await asyncio.gather(*tasks)



async def _process_one(att: dict) -> dict:
    try:
        if att["type"] == "IMAGE":
            text = await process_image(att["url"], att.get("originalName", "image"))
        elif att["type"] == "AUDIO":
            text = await process_audio(att["url"], att.get("originalName", "audio"))
        else:
            text = "[unknown attachment type]"
    except Exception as e:
        print(f"[RAQIFY] Attachment failed ({att.get('originalName')}): {e}")
        text = "[failed to process attachment]"

    return {
        "type": att["type"],
        "name": att.get("originalName", ""),
        "extractedText": text,
    }


async def process_image(url: str, name: str) -> str:
    async with httpx.AsyncClient() as http:
        r = await http.post(IMAGE_EXTRACTION_API, json={"url": url}, timeout=30)
        r.raise_for_status()
    return r.json().get("processed_text", "[no text extracted]")


async def process_audio(url: str, name: str) -> str:
    async with httpx.AsyncClient() as http:
        r = await http.post(AUDIO_EXTRACTION_API, params={"audio_url": url}, timeout=120)
        r.raise_for_status()
    return r.json().get("processed_text", "[no text extracted]")