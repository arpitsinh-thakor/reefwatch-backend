import base64
import csv
import io
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether

from inference import CLASSES, MODEL_NAMES, CoralPredictor
from segmentation import SegmentationPredictor

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "reefwatch.sqlite3"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_CSV_BYTES = 20 * 1024 * 1024
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Reefwatch Coral Research API",
    description="Notebook-faithful coral classification, Grad-CAM, review queue, research evaluation, exports, and reef observations.",
    version="2.0.0",
)
origins = [x.strip() for x in os.getenv("CORAL_FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST", "PATCH"], allow_headers=["*"])

app.mount("/media", StaticFiles(directory=IMAGE_DIR), name="media")

predictor: Optional[CoralPredictor] = None
model_load_error: Optional[str] = None
segmentation_predictor: Optional[SegmentationPredictor] = None
segmentation_load_error: Optional[str] = None


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL, confidence REAL NOT NULL, probabilities TEXT NOT NULL,
            model TEXT NOT NULL, checkpoint TEXT NOT NULL, device TEXT NOT NULL, source TEXT NOT NULL,
            image_path TEXT NOT NULL, corrected_path TEXT, heatmap_path TEXT,
            top_two_margin REAL NOT NULL, normalized_entropy REAL NOT NULL, needs_review INTEGER NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'pending', review_note TEXT NOT NULL DEFAULT '',
            reef_id TEXT NOT NULL DEFAULT '', observed_at TEXT NOT NULL, location TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '', manual_label TEXT NOT NULL DEFAULT '', tta_views INTEGER NOT NULL DEFAULT 3,
            preprocessing TEXT NOT NULL DEFAULT '',
            reproducibility_json TEXT NOT NULL DEFAULT '{}'
        )""")
        # Backward-compatible migration for databases created by earlier Reefwatch versions.
        existing = {row[1] for row in con.execute("PRAGMA table_info(predictions)").fetchall()}
        if "reproducibility_json" not in existing:
            con.execute("ALTER TABLE predictions ADD COLUMN reproducibility_json TEXT NOT NULL DEFAULT '{}'")
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at DESC)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_reef ON predictions(reef_id, observed_at)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_review ON predictions(review_status, needs_review)")


init_db()


@app.on_event("startup")
def load_model():
    global predictor, model_load_error, segmentation_predictor, segmentation_load_error
    try:
        predictor = CoralPredictor(CHECKPOINT_DIR, fold=int(os.getenv("REEFWATCH_FOLD", "0")))
        model_load_error = None
    except Exception as exc:
        predictor = None
        model_load_error = str(exc)
    seg_path = CHECKPOINT_DIR / "coral_damage_segmentation.ts"
    if seg_path.is_file():
        try:
            segmentation_predictor = SegmentationPredictor(seg_path)
            segmentation_load_error = None
        except Exception as exc:
            segmentation_predictor = None
            segmentation_load_error = str(exc)
    else:
        segmentation_predictor = None
        segmentation_load_error = "No TorchScript segmentation checkpoint found; annotated masks and model training/export are required."


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None, "model": "EfficientNetV2-S + ConvNeXt-Tiny ensemble + 3-view TTA", "fold": int(os.getenv("REEFWATCH_FOLD", "0")), "classes": CLASSES, "error": model_load_error}


@app.get("/sources")
def sources():
    return {"sources": ["unspecified", "bhd_corals", "bleached_unbleached", "healthy_bleached_flickr", "survey_a", "survey_b", "flickr", "kaggle", "other"], "note": "Source is observation metadata. It does not switch model weights."}


def _decode_image(content: bytes):
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 15 MB or smaller.")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")
    return image


def _store_result(filename: str, raw_content: bytes, result: dict, source: str, reef_id: str, observed_at: str, location: str, notes: str) -> dict:
    pid = str(uuid.uuid4())
    image_name = f"{pid}.jpg"
    image_path = IMAGE_DIR / image_name
    original = cv2.imdecode(np.frombuffer(raw_content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError("Image decode failed while saving prediction history.")
    cv2.imwrite(str(image_path), original, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    corrected_path = None
    heatmap_path = None
    for key, suffix, target in [("corrected_image", "-corrected.jpg", "corrected"), ("gradcam_heatmap", "-gradcam.jpg", "heatmap")]:
        value = result.get(key)
        if value and "," in value:
            raw = base64.b64decode(value.split(",", 1)[1])
            path = IMAGE_DIR / f"{pid}{suffix}"
            path.write_bytes(raw)
            if target == "corrected": corrected_path = str(path.relative_to(BASE_DIR))
            else: heatmap_path = str(path.relative_to(BASE_DIR))
    needs_review = bool(result.get("needs_review", False))
    created = now_iso()
    review_status = "needs_review" if needs_review else "pending"
    probs = result.get("probabilities", {})
    with db() as con:
        con.execute("""INSERT INTO predictions
        (id,created_at,filename,predicted_class,confidence,probabilities,model,checkpoint,device,source,
         image_path,corrected_path,heatmap_path,top_two_margin,normalized_entropy,needs_review,review_status,
         reef_id,observed_at,location,notes,tta_views,preprocessing,reproducibility_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (pid, created, Path(filename).name[:255], result["predicted_class"], float(result["confidence"]), json.dumps(probs),
         result.get("model", ""), result.get("checkpoint", ""), result.get("device", ""), source,
         str(image_path.relative_to(BASE_DIR)), corrected_path, heatmap_path,
         float(result.get("top_two_margin", 0)), float(result.get("normalized_entropy", 1)), int(needs_review), review_status,
         reef_id.strip()[:120], observed_at or created, location.strip()[:255], notes.strip()[:2000], int(result.get("tta_views", 3)), result.get("preprocessing", ""), json.dumps({"model": result.get("model", ""), "backbones": result.get("backbones", MODEL_NAMES), "checkpoint": result.get("checkpoint", ""), "fold": result.get("fold", 0), "class_order": CLASSES, "preprocessing": result.get("preprocessing", ""), "tta_views": result.get("tta_views", 3), "tta_transforms": ["identity", "horizontal_flip", "RandomRotate90"], "ensemble": "mean of softmax probabilities across 2 backbones x 3 views", "image_size": 224, "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "source": source, "protocol": "single-image inference; not a CV/LOSO evaluation"})))
    result.update({"id": pid, "created_at": created, "filename": Path(filename).name, "image_url": f"/media/{image_name}",
                   "corrected_url": f"/media/{pid}-corrected.jpg" if corrected_path else None,
                   "heatmap_url": f"/media/{pid}-gradcam.jpg" if heatmap_path else None,
                   "review_status": review_status, "reef_id": reef_id, "observed_at": observed_at or created,
                   "location": location, "notes": notes})
    # Do not retain large data URIs in the SQLite row/API history list; immediate response may still use them.
    return result


@app.post("/predict")
async def predict(file: UploadFile = File(...), source: str = Form("unspecified"), explain: bool = Form(True),
                  reef_id: str = Form(""), observed_at: str = Form(""), location: str = Form(""), notes: str = Form("")):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Notebook ensemble not loaded. Add both fold-0 checkpoints listed in README and restart. " + (model_load_error or ""))
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    content = await file.read()
    image = _decode_image(content)
    try:
        result = predictor.predict_bgr(image, source=source, explain=explain)
        return _store_result(file.filename or "coral-image.jpg", content, result, source, reef_id, observed_at, location, notes)
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), source: str = Form("unspecified"), explain: bool = Form(False), reef_id: str = Form(""), observed_at: str = Form(""), location: str = Form(""), notes: str = Form("")):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Notebook ensemble not loaded. Add both fold-0 checkpoints and restart.")
    if not files or len(files) > 100:
        raise HTTPException(status_code=400, detail="Upload between 1 and 100 images per batch.")
    items, errors = [], []
    for f in files:
        try:
            if f.content_type and not f.content_type.startswith("image/"):
                raise ValueError("not an image upload")
            content = await f.read()
            image = _decode_image(content)
            result = predictor.predict_bgr(image, source=source, explain=explain)
            items.append(_store_result(f.filename or "coral-image.jpg", content, result, source, reef_id, observed_at, location, notes))
        except Exception as exc:
            errors.append({"filename": f.filename or "unknown", "error": str(exc)})
    return {"count": len(items), "failed": len(errors), "items": items, "errors": errors}


def _row_to_item(row):
    d = dict(row)
    d["probabilities"] = json.loads(d.get("probabilities") or "{}")
    d["needs_review"] = bool(d.get("needs_review"))
    try: d["reproducibility"] = json.loads(d.get("reproducibility_json") or "{}")
    except (TypeError, json.JSONDecodeError): d["reproducibility"] = {}
    for field, url_field, suffix in [("image_path", "image_url", ""), ("corrected_path", "corrected_url", "-corrected.jpg"), ("heatmap_path", "heatmap_url", "-gradcam.jpg")]:
        path = d.get(field)
        if path:
            name = Path(path).name
            d[url_field] = f"/media/{name}"
        else:
            d[url_field] = None
    return d


@app.get("/history")
def history(q: str = "", source: str = "", predicted_class: str = "", review_status: str = "", reef_id: str = "", limit: int = Query(100, ge=1, le=500)):
    sql = "SELECT * FROM predictions WHERE 1=1"
    args = []
    if q.strip():
        term = f"%{q.strip()}%"
        sql += " AND (filename LIKE ? OR notes LIKE ? OR location LIKE ? OR reef_id LIKE ? OR id LIKE ?)"
        args += [term, term, term, term, term]
    if source: sql += " AND source=?"; args.append(source)
    if predicted_class in CLASSES: sql += " AND predicted_class=?"; args.append(predicted_class)
    if review_status: sql += " AND review_status=?"; args.append(review_status)
    if reef_id.strip(): sql += " AND reef_id=?"; args.append(reef_id.strip())
    sql += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
    with db() as con:
        rows = con.execute(sql, args).fetchall()
    return {"items": [_row_to_item(r) for r in rows], "count": len(rows)}


@app.get("/predictions/{prediction_id}")
def history_detail(prediction_id: str):
    with db() as con:
        row = con.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,)).fetchone()
    if not row: raise HTTPException(status_code=404, detail="Prediction not found.")
    return _row_to_item(row)


class ReviewUpdate(BaseModel):
    review_status: str = Field(pattern="^(pending|needs_review|confirmed|rejected)$")
    manual_label: str = ""
    review_note: str = ""


@app.patch("/predictions/{prediction_id}/review")
def update_review(prediction_id: str, update: ReviewUpdate):
    if update.manual_label and update.manual_label not in CLASSES:
        raise HTTPException(status_code=422, detail=f"manual_label must be one of {CLASSES}")
    with db() as con:
        cur = con.execute("UPDATE predictions SET review_status=?, manual_label=?, review_note=? WHERE id=?",
                           (update.review_status, update.manual_label, update.review_note[:2000], prediction_id))
        if cur.rowcount == 0: raise HTTPException(status_code=404, detail="Prediction not found.")
        row = con.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,)).fetchone()
    return _row_to_item(row)


def _filtered_rows(source: str = "", review_status: str = "", reef_id: str = ""):
    sql = "SELECT * FROM predictions WHERE 1=1"; args = []
    if source: sql += " AND source=?"; args.append(source)
    if review_status: sql += " AND review_status=?"; args.append(review_status)
    if reef_id: sql += " AND reef_id=?"; args.append(reef_id)
    sql += " ORDER BY created_at DESC"
    with db() as con: return con.execute(sql, args).fetchall()


@app.get("/history/export.csv")
def export_history_csv(source: str = "", review_status: str = "", reef_id: str = ""):
    rows = _filtered_rows(source, review_status, reef_id)
    output = io.StringIO()
    fields = ["id", "created_at", "filename", "predicted_class", "confidence", "probabilities", "source", "model", "checkpoint", "reef_id", "observed_at", "location", "top_two_margin", "normalized_entropy", "needs_review", "review_status", "manual_label", "review_note", "notes"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        d = dict(r); d["probabilities"] = d.get("probabilities", "{}"); writer.writerow(d)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=reefwatch-prediction-history.csv"})


def _pdf_for_prediction(row):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17675d")))
    story = [Paragraph("Reefwatch — Coral Classification Report", styles["ReportTitle"]), Spacer(1, 5*mm)]
    story.append(Paragraph("Research prototype report. Automated predictions require expert/field verification.", styles["BodyText"]))
    story.append(Spacer(1, 4*mm))
    probabilities = json.loads(row["probabilities"] or "{}")
    details = [
        ["Prediction ID", row["id"]], ["File", row["filename"]], ["Observation time", row["observed_at"]],
        ["Predicted class", row["predicted_class"].upper()], ["Top-class score", f"{row['confidence']:.4f}"],
        ["Source", row["source"]], ["Model", row["model"]], ["Checkpoint", row["checkpoint"]],
        ["Fold / TTA", f"fold {os.getenv('REEFWATCH_FOLD','0')} / {row['tta_views']} views"],
        ["Reef ID", row["reef_id"] or "—"], ["Location", row["location"] or "—"],
        ["Review status", row["review_status"]], ["Manual label", row["manual_label"] or "—"],
        ["Class scores", " | ".join(f"{k}: {float(v):.4f}" for k, v in probabilities.items())],
        ["Uncertainty", f"margin={row['top_two_margin']:.4f}; normalized entropy={row['normalized_entropy']:.4f}"],
        ["Notes", row["notes"] or "—"], ["Reviewer note", row["review_note"] or "—"],
    ]
    table = Table(details, colWidths=[42*mm, 125*mm])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(0,-1),colors.HexColor("#edf5f3")),("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#d6e3e0")),("VALIGN",(0,0),(-1,-1),"TOP"),("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
    story.extend([table, Spacer(1, 5*mm)])
    for key, title in [("image_path", "Original image"), ("corrected_path", "CLAHE-corrected image"), ("heatmap_path", "Grad-CAM heatmap")]:
        path = BASE_DIR / row[key] if row.get(key) else None
        if path and path.exists():
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(RLImage(str(path), width=78*mm, height=58*mm, kind="proportional"))
            story.append(Spacer(1, 3*mm))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("Preprocessing: " + (row["preprocessing"] or "Notebook preprocessing"), styles["BodyText"]))
    doc.build(story)
    buffer.seek(0)
    return buffer


@app.get("/predictions/{prediction_id}/report.pdf")
def prediction_pdf(prediction_id: str):
    with db() as con: row = con.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,)).fetchone()
    if not row: raise HTTPException(status_code=404, detail="Prediction not found.")
    pdf = _pdf_for_prediction(dict(row))
    return StreamingResponse(pdf, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=reefwatch-{prediction_id[:8]}.pdf"})


@app.get("/history/export.pdf")
def export_history_pdf(source: str = "", review_status: str = "", reef_id: str = ""):
    rows = [dict(r) for r in _filtered_rows(source, review_status, reef_id)]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=12*mm, leftMargin=12*mm, topMargin=14*mm, bottomMargin=14*mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("Reefwatch — Batch Prediction Summary", styles["Title"]), Spacer(1, 5*mm), Paragraph(f"Records: {len(rows)} · Generated {now_iso()}", styles["BodyText"]), Spacer(1, 4*mm)]
    data = [["Time", "File", "Source", "Reef", "Prediction", "Score", "Review"]]
    for r in rows:
        data.append([r["created_at"][:19], r["filename"][:28], r["source"][:20], r["reef_id"][:16], r["predicted_class"], f"{r['confidence']:.3f}", r["review_status"]])
    table = Table(data, repeatRows=1, colWidths=[27*mm, 39*mm, 29*mm, 22*mm, 22*mm, 15*mm, 24*mm])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17675d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.25,colors.HexColor("#cbd8d5")),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f3f8f7")])]))
    story.append(table)
    doc.build(story); buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=reefwatch-batch-report.pdf"})


# -- Evaluation: use ground-truth labels from exported per-image predictions. --
def _norm_label(value):
    s = str(value or "").strip().lower()
    # Notebook label indices: 0=bleached, 1=dead, 2=healthy.
    if s in {"0", "0.0"}: return "bleached"
    if s in {"1", "1.0"}: return "dead"
    if s in {"2", "2.0"}: return "healthy"
    return s


def metrics_for(rows):
    matrix = {actual: {pred: 0 for pred in CLASSES} for actual in CLASSES}
    for actual, pred in rows:
        if actual in matrix and pred in matrix[actual]: matrix[actual][pred] += 1
    total = len(rows); correct = sum(matrix[c][c] for c in CLASSES)
    per_class = {}
    for c in CLASSES:
        tp = matrix[c][c]
        fp = sum(matrix[a][c] for a in CLASSES if a != c)
        fn = sum(matrix[c][p] for p in CLASSES if p != c)
        support = sum(matrix[c].values())
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"samples": total, "accuracy": correct/total if total else 0.0, "macro_precision": sum(v["precision"] for v in per_class.values())/len(CLASSES), "macro_recall": sum(v["recall"] for v in per_class.values())/len(CLASSES), "macro_f1": sum(v["f1"] for v in per_class.values())/len(CLASSES), "confusion_matrix": matrix, "classes": CLASSES, "per_class": per_class}


def _read_csv(raw: bytes):
    try: return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error) as exc: raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_CSV_BYTES: raise HTTPException(status_code=413, detail="CSV must be 20 MB or smaller.")
    rows = _read_csv(raw)
    if not rows: raise HTTPException(status_code=400, detail="CSV has no data rows.")
    aliases_true = ["true_label", "label", "y_true", "actual_label", "ground_truth"]
    aliases_pred = ["predicted_label", "pred_ensemble_label", "pred_label", "predicted_class", "pred", "y_pred"]
    def first(row, aliases):
        lowered = {str(k or "").strip().lower(): v for k,v in row.items()}
        for key in aliases:
            if key in lowered and str(lowered[key] or "").strip() != "": return lowered[key]
        return ""
    records, excluded = [], 0
    for row in rows:
        actual, pred = _norm_label(first(row, aliases_true)), _norm_label(first(row, aliases_pred))
        if actual not in CLASSES or pred not in CLASSES:
            excluded += 1; continue
        records.append({"true": actual, "pred": pred, "source": str(first(row,["source","dataset","domain"]) or "unspecified").strip(),
                        "model": str(first(row,["model","backbone","variant"]) or "unspecified").strip(),
                        "protocol": str(first(row,["protocol","evaluation_protocol","split_type","evaluation_type"]) or "unspecified").strip().lower(),
                        "fold": str(first(row,["fold","fold_id"]) or "").strip()})
    if not records: raise HTTPException(status_code=400, detail="No valid per-image prediction rows. Include true labels and predictions using bleached/dead/healthy or notebook class indices 0/1/2.")
    overall = metrics_for([(r["true"],r["pred"]) for r in records])
    def group(key):
        result = {}
        for name in sorted({r[key] for r in records}):
            subset = [r for r in records if r[key] == name]
            result[name] = metrics_for([(r["true"],r["pred"]) for r in subset])
        return result
    by_source, by_model, by_protocol = group("source"), group("model"), group("protocol")
    source_protocol = {}
    for r in records:
        key = f"{r['source']} | {r['protocol']}"
        source_protocol.setdefault(key, []).append((r["true"],r["pred"]))
    return {"overall": overall, "by_source": by_source, "by_model": by_model, "by_protocol": by_protocol,
            "by_source_protocol": {k:metrics_for(v) for k,v in source_protocol.items()}, "valid_rows":len(records),
            "excluded_rows":excluded, "label_order":CLASSES,
            "csv_schema":"true_label,predicted_label,source,model,protocol,fold",
            "protocol_note":"CV should describe within-source/cross-validation predictions; LOSO should describe train-on-other-sources, test-on-held-out-source predictions. The API does not infer protocol from source names."}


@app.post("/evaluate/cross-dataset")
async def evaluate_cross_dataset(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_CSV_BYTES: raise HTTPException(status_code=413, detail="CSV must be 20 MB or smaller.")
    rows = _read_csv(raw)
    if not rows: raise HTTPException(status_code=400, detail="CSV has no data rows.")
    normalized = []
    for row in rows:
        r = {str(k or "").strip().lower(): v for k,v in row.items()}
        source = r.get("source", "")
        if not source or not ("in_domain_acc" in r and "cross_dataset_acc" in r): continue
        try:
            ind, cross = float(r["in_domain_acc"]), float(r["cross_dataset_acc"])
            if ind > 1: ind /= 100
            if cross > 1: cross /= 100
            gap = float(r.get("generalization_gap_pp") or ((ind-cross)*100))
            normalized.append({"source": source, "in_domain_accuracy": ind, "cross_dataset_accuracy": cross,
                               "gap_percentage_points": gap,
                               "in_domain_ci_low": float(r.get("in_domain_ci_lo") or 0), "in_domain_ci_high": float(r.get("in_domain_ci_hi") or 0),
                               "cross_dataset_ci_low": float(r.get("cross_dataset_ci_lo") or 0), "cross_dataset_ci_high": float(r.get("cross_dataset_ci_hi") or 0)})
        except (ValueError, TypeError): continue
    if not normalized:
        raise HTTPException(status_code=400, detail="Expected notebook generalization_gap_table.csv columns: source, in_domain_acc, cross_dataset_acc, optional CI/gap columns.")
    return {"protocol":"in-domain vs leave-one-source-out cross-dataset", "rows":normalized, "count":len(normalized), "source":"uploaded notebook experiment CSV"}


@app.get("/error-analysis")
def error_analysis(source: str = "", limit: int = Query(100, ge=1, le=500)):
    """Return expert-labelled history records where the model prediction disagrees with the manual label.

    Only reviewed records with an explicit manual_label are included; prediction confidence alone is not ground truth.
    """
    sql = "SELECT * FROM predictions WHERE manual_label != '' AND manual_label IN ('bleached','dead','healthy') AND manual_label != predicted_class"
    args = []
    if source:
        sql += " AND source=?"; args.append(source)
    sql += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
    with db() as con:
        rows = con.execute(sql, args).fetchall()
    items = [_row_to_item(row) for row in rows]
    return {"items": items, "count": len(items), "source": source or "all sources",
            "basis": "Only records with an explicit manual_label different from predicted_class; not inferred from confidence."}


@app.get("/predictions/{prediction_id}/reproducibility")
def prediction_reproducibility(prediction_id: str):
    with db() as con:
        row = con.execute("SELECT id, filename, created_at, reproducibility_json FROM predictions WHERE id=?", (prediction_id,)).fetchone()
    if not row: raise HTTPException(status_code=404, detail="Prediction record not found.")
    try: config = json.loads(row["reproducibility_json"] or "{}")
    except json.JSONDecodeError: config = {}
    return {"prediction_id": row["id"], "filename": row["filename"], "created_at": row["created_at"], "configuration": config}


@app.get("/monitoring")
def monitoring(reef_id: str = "", limit: int = Query(200, ge=1, le=1000)):
    sql = "SELECT * FROM predictions WHERE reef_id<>''"; args=[]
    if reef_id.strip(): sql += " AND reef_id=?"; args.append(reef_id.strip())
    sql += " ORDER BY observed_at ASC LIMIT ?"; args.append(limit)
    with db() as con: rows=con.execute(sql,args).fetchall()
    items=[_row_to_item(r) for r in rows]
    grouped=defaultdict(lambda:{"observations":0,"class_counts":{c:0 for c in CLASSES},"mean_confidence":0.0})
    for item in items:
        key=item["reef_id"] or "unspecified"
        grouped[key]["observations"]+=1
        grouped[key]["class_counts"][item["predicted_class"]]+=1
        grouped[key]["mean_confidence"]+=float(item["confidence"])
    for v in grouped.values(): v["mean_confidence"]/=max(v["observations"],1)
    return {"items":items,"count":len(items),"by_reef":dict(grouped),"note":"Trend comparisons are only meaningful for comparable sites, viewpoints, acquisition conditions, and validated labels. Predictions alone are not direct measurements of coral cover or ecological change."}


@app.get("/segmentation/status")
def segmentation_status():
    path = CHECKPOINT_DIR / "coral_damage_segmentation.ts"
    return {"available": segmentation_predictor is not None, "checkpoint_expected": str(path.relative_to(BASE_DIR)), "status": "ready" if segmentation_predictor is not None else "not_configured", "message": "Optional TorchScript contract: input [1,3,224,224] after notebook-compatible CLAHE/ImageNet normalization; output raw logits [1,1,H,W] or [1,2,H,W] (background, damage). No segmentation model or pixel-mask labels are present in the uploaded notebook; Grad-CAM is not a segmentation mask.", "error": segmentation_load_error}

@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    if segmentation_predictor is None:
        status = segmentation_status()
        raise HTTPException(status_code=501, detail=status["message"] + " " + str(status.get("error") or ""))
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")
    content = await file.read()
    image = _decode_image(content)
    try:
        return segmentation_predictor.predict_bgr(image)
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Segmentation failed: {exc}") from exc
