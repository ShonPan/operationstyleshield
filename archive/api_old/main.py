"""
StyleShield API — upload CSV, run scorer, return JSON for React UI.
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import traceback

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load scorer from project root (filename not a standard package name on all OS)
import importlib.util

_spec = importlib.util.spec_from_file_location("styleshield_module", ROOT / "Styleshield_script.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
StyleShieldScorer = _mod.StyleShieldScorer

app = FastAPI(title="StyleShield API")


@app.exception_handler(RequestValidationError)
async def validation_handler(_, exc: RequestValidationError):
    """422 → same shape as UI expects (e.g. missing multipart field 'file')."""
    return JSONResponse(
        status_code=200,
        content={
            "ok": False,
            "error": "Invalid request: upload a CSV using field name 'file'. "
            + str(exc.errors()[:2] if exc.errors() else exc),
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # optional fixed origin
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    return obj


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    tmp = None
    suffix = Path(file.filename or "upload.csv").suffix or ".csv"
    if suffix.lower() != ".csv":
        suffix = ".csv"
    buf = await file.read()
    if not buf:
        return {"ok": False, "error": "Empty file"}

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(buf)
        tmp.close()
        path = tmp.name

        log = io.StringIO()
        with redirect_stdout(log):
            scorer = StyleShieldScorer()
            results_df, clusters = scorer.analyze_csv(path)

        cols = [
            "account_id",
            "confidence",
            "bot_score",
            "cluster_id",
            "vocabulary_uniformity",
            "structural_regularity",
            "hedging_signature",
        ]
        for c in cols:
            if c not in results_df.columns:
                results_df[c] = None
        top15 = results_df[cols].head(15)
        rows = json.loads(top15.to_json(orient="records", double_precision=6))

        return {
            "ok": True,
            "log": log.getvalue(),
            "top15": rows,
            "clusters": _json_safe(clusters),
            "total_accounts": int(len(results_df)),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e) or type(e).__name__,
            "detail": traceback.format_exc()[-4000:],  # helps debug CSV / deps issues
        }
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
