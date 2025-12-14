from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import time
import json
import joblib
import numpy as np
import os

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# -------------------------
# Tracing setup
# -------------------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# -------------------------
# Structured logging
# -------------------------
logger = logging.getLogger("heart-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Heart Disease Prediction API")

# Application state
app_state = {
    "is_ready": False,
    "is_alive": True,
    "model": None
}

# -------------------------
# Input schema
# -------------------------
class Input(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# -------------------------
# Startup: Load model
# -------------------------
@app.on_event("startup")
async def startup_event():
    try:
        model_path = os.getenv("MODEL_PATH", "heart_model.joblib")
        logger.info(f"Loading model from {model_path}...")
        app_state["model"] = joblib.load(model_path)
        app_state["is_ready"] = True
        logger.info("Model loaded successfully")
    except Exception as e:
        app_state["is_ready"] = False
        logger.exception(f"Failed to load model: {e}")


# -------------------------
# Health probes
# -------------------------
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


# -------------------------
# Middleware: latency
# -------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response


# -------------------------
# Global exception handler
# -------------------------
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")

    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )


# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict", tags=["Prediction"])
async def predict(input: Input, request: Request):

    if not app_state["is_ready"]:
        raise HTTPException(status_code=503, detail="Model not ready")

    model = app_state["model"]

    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            input_data = input.dict()

            features = np.array([[
                input_data["age"],
                input_data["sex"],
                input_data["cp"],
                input_data["trestbps"],
                input_data["chol"],
                input_data["fbs"],
                input_data["restecg"],
                input_data["thalach"],
                input_data["exang"],
                input_data["oldpeak"],
                input_data["slope"],
                input_data["ca"],
                input_data["thal"],
            ]])

            # Prediction
            pred = int(model.predict(features)[0])

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(features)[0]
                confidence = round(float(proba[1]), 4)

            latency = round((time.time() - start_time) * 1000, 2)

            result = {
                "prediction": pred,
                "confidence": confidence,
                "trace_id": trace_id,
                "latency_ms": latency
            }

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data,
                "result": result,
                "status": "success"
            }))

            return result

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
