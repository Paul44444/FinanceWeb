from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json
import threading
from queue import Queue

from fastapi.responses import StreamingResponse

from backend.main import net_1

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://financeweb-three.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CalculationInput(BaseModel):
    x: float
    y: float

@app.get("/api/run-network-stream")
def run_network_stream():
    def event_stream():
        messages = Queue()

        def send_progress(progress):
            messages.put({
                "type": "progress",
                **progress,
            })

        def run_training():
            try:
                result = net_1(progress_callback=send_progress)

                messages.put({
                    "type": "complete",
                    "data": result,
                })
            except Exception as error:
                messages.put({
                    "type": "error",
                    "message": str(error),
                })
            finally:
                messages.put(None)

        worker = threading.Thread(
            target=run_training,
            daemon=True,
        )
        worker.start()

        while True:
            message = messages.get()

            if message is None:
                break

            yield f"data: {json.dumps(message)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/")
def root():
    return {
        "message": "FastAPI backend is running"
    }

@app.get("/api/hello")
def hello():
    return {
        "message": "Hello from the Python backend!"
    }

@app.post("/api/calculate")
def calculate(values: CalculationInput):
    result = values.x ** 2 + values.y ** 2

    return {
        "result": result
    }

@app.post("/api/run-network")
def run_network():
    try:
        result = net_1()

        return {
            "success": True,
            "data": result,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error