from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Any

from brief_service import handle_brief_request

app = FastAPI(title="RAQIFY AI Service")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/brief")
async def brief_endpoint(body: dict[str, Any]) -> Any:  # ← بدل Request
    if "type" not in body:
        raise HTTPException(status_code=400, detail="Missing field: 'type'")

    if body["type"] not in {"INITIAL_BRIEF", "REFINEMENT"}:
        raise HTTPException(status_code=400, detail=f"Invalid type: {body['type']}")

    try:
        result = await handle_brief_request(body)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[RAQIFY] Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")