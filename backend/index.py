from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import json
import threading
from queue import Queue

from fastapi.responses import StreamingResponse

from backend.main import net_1
from backend.electricity import train_electricity_forecast

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


class ElectricityTrainingInput(BaseModel):
    lookback_days: int = 60
    iterations: int = 800
    forecast_hours: int = 48
    starting_budget: float = 1000.0
    storage_capacity_mwh: float = 1.0
    round_trip_efficiency: float = 0.90
    charge_power_mw: float = 0.5
    battery_cost_per_kwh: float = 400.0
    market_fee_per_mwh: float = 3.0
    degradation_cost_per_mwh: float = 20.0
    self_discharge_percent_per_day: float = 0.2


@app.post("/api/electricity/train")
def train_electricity(values: ElectricityTrainingInput):
    if not 21 <= values.lookback_days <= 365:
        raise HTTPException(status_code=422, detail="lookback_days must be 21 to 365")
    if not 50 <= values.iterations <= 5000:
        raise HTTPException(status_code=422, detail="iterations must be 50 to 5000")
    if not 12 <= values.forecast_hours <= 72:
        raise HTTPException(status_code=422, detail="forecast_hours must be 12 to 72")
    if not 100 <= values.starting_budget <= 1_000_000:
        raise HTTPException(status_code=422, detail="starting_budget must be 100 to 1000000")
    if not 0.01 <= values.storage_capacity_mwh <= 100:
        raise HTTPException(status_code=422, detail="storage_capacity_mwh must be 0.01 to 100")
    if not 0.5 <= values.round_trip_efficiency <= 1:
        raise HTTPException(status_code=422, detail="round_trip_efficiency must be 0.5 to 1")
    if not 0.01 <= values.charge_power_mw <= 100:
        raise HTTPException(status_code=422, detail="charge_power_mw must be 0.01 to 100")
    if not 0 <= values.battery_cost_per_kwh <= 5000:
        raise HTTPException(status_code=422, detail="battery_cost_per_kwh must be 0 to 5000")
    if not 0 <= values.market_fee_per_mwh <= 500:
        raise HTTPException(status_code=422, detail="market_fee_per_mwh must be 0 to 500")
    if not 0 <= values.degradation_cost_per_mwh <= 1000:
        raise HTTPException(status_code=422, detail="degradation_cost_per_mwh must be 0 to 1000")
    if not 0 <= values.self_discharge_percent_per_day <= 20:
        raise HTTPException(status_code=422, detail="self_discharge_percent_per_day must be 0 to 20")
    try:
        return train_electricity_forecast(
            values.lookback_days,
            values.iterations,
            values.forecast_hours,
            values.starting_budget,
            values.storage_capacity_mwh,
            values.round_trip_efficiency,
            values.charge_power_mw,
            values.battery_cost_per_kwh,
            values.market_fee_per_mwh,
            values.degradation_cost_per_mwh,
            values.self_discharge_percent_per_day,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

@app.get("/api/run-network-stream")
def run_network_stream(
    iterations: int = Query(default=200, ge=1, le=10000),
):
    def event_stream():
        messages = Queue()

        def send_progress(progress):
            messages.put({
                "type": "progress",
                **progress,
            })

        def run_training():
            try:
                result = net_1(
                    progress_callback=send_progress,
                    training_steps=iterations,
                )

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
def run_network(iterations: int = Query(default=200, ge=1, le=10000)):
    try:
        result = net_1(training_steps=iterations)

        return {
            "success": True,
            "data": result,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
