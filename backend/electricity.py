from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from threading import Lock
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch import nn


SMARD_BASE = "https://www.smard.de/app/chart_data/4169/DE-LU"
ARCHIVE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARIABLES = (
    "temperature_2m,wind_speed_100m,shortwave_radiation,cloud_cover"
)
GERMANY_LOCATIONS = {
    "Berlin": (52.52, 13.41),
    "Hamburg": (53.55, 9.99),
    "Munich": (48.14, 11.58),
    "Frankfurt": (50.11, 8.68),
}

_training_lock = Lock()
_latest_model: dict | None = None


class ElectricityPriceNetwork(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_count, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def _fetch_json(url: str, params: dict | None = None):
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "finance1-energy-research/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _iso_to_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _ms_to_iso(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()


def _fetch_smard_prices(lookback_days: int) -> dict[int, float]:
    index = _fetch_json(f"{SMARD_BASE}/index_hour.json")
    earliest = int(
        (datetime.now(timezone.utc) - timedelta(days=lookback_days + 9)).timestamp()
        * 1000
    )
    timestamps = [
        value for value in index["timestamps"] if value >= earliest
    ]
    prices: dict[int, float] = {}
    for timestamp in timestamps:
        payload = _fetch_json(
            f"{SMARD_BASE}/4169_DE-LU_hour_{timestamp}.json"
        )
        for point_timestamp, price in payload.get("series", []):
            if price is not None:
                prices[int(point_timestamp)] = float(price)
    return prices


def _weather_params(start_date: str, end_date: str) -> dict[str, str]:
    latitudes = ",".join(str(value[0]) for value in GERMANY_LOCATIONS.values())
    longitudes = ",".join(str(value[1]) for value in GERMANY_LOCATIONS.values())
    return {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": WEATHER_VARIABLES,
        "timezone": "GMT",
    }


def _average_weather(payload) -> dict[int, dict[str, float]]:
    locations = payload if isinstance(payload, list) else [payload]
    combined: dict[int, dict[str, list[float]]] = {}
    for location in locations:
        hourly = location["hourly"]
        for index, time_value in enumerate(hourly["time"]):
            timestamp = _iso_to_ms(time_value)
            bucket = combined.setdefault(
                timestamp,
                {name: [] for name in WEATHER_VARIABLES.split(",")},
            )
            for name in bucket:
                value = hourly[name][index]
                if value is not None:
                    bucket[name].append(float(value))

    return {
        timestamp: {
            name: float(np.mean(values)) if values else 0.0
            for name, values in variables.items()
        }
        for timestamp, variables in combined.items()
    }


def _fetch_historical_weather(
    start: datetime,
    end: datetime,
) -> dict[int, dict[str, float]]:
    payload = _fetch_json(
        ARCHIVE_WEATHER_URL,
        _weather_params(start.date().isoformat(), end.date().isoformat()),
    )
    return _average_weather(payload)


def _fetch_forecast_weather(forecast_days: int) -> dict[int, dict[str, float]]:
    start = datetime.now(timezone.utc).date() - timedelta(days=1)
    end = start + timedelta(days=forecast_days + 1)
    payload = _fetch_json(
        FORECAST_WEATHER_URL,
        _weather_params(start.isoformat(), end.isoformat()),
    )
    return _average_weather(payload)


def _calendar_features(timestamp: int) -> list[float]:
    date = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
    hour_angle = 2 * math.pi * date.hour / 24
    weekday_angle = 2 * math.pi * date.weekday() / 7
    year_angle = 2 * math.pi * date.timetuple().tm_yday / 365.25
    return [
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(weekday_angle),
        math.cos(weekday_angle),
        math.sin(year_angle),
        math.cos(year_angle),
    ]


def _feature_vector(
    timestamp: int,
    weather: dict[str, float],
    price_24h: float,
    price_168h: float,
) -> list[float]:
    return [
        *_calendar_features(timestamp),
        weather["temperature_2m"],
        weather["wind_speed_100m"],
        weather["shortwave_radiation"],
        weather["cloud_cover"],
        price_24h,
        price_168h,
    ]


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    correlation = 0.0
    if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(actual, predicted)[0, 1])
    return {"mae": mae, "rmse": rmse, "correlation": correlation}


def _best_daily_cycles(points: list[dict], efficiency: float) -> list[dict]:
    leg_efficiency = math.sqrt(efficiency)
    cycles = []
    for start in range(0, len(points), 24):
        day = points[start : start + 24]
        best = None
        for buy_index in range(len(day) - 1):
            for sell_index in range(buy_index + 1, len(day)):
                expected_margin = (
                    day[sell_index]["predicted"] * leg_efficiency
                    - day[buy_index]["predicted"] / leg_efficiency
                    - 2.0
                )
                if best is None or expected_margin > best["expectedMargin"]:
                    best = {
                        "buyIndex": start + buy_index,
                        "sellIndex": start + sell_index,
                        "expectedMargin": expected_margin,
                    }
        if best is not None and best["expectedMargin"] > 0:
            cycles.append(best)
    return cycles


def _simulate_battery_policy(
    validation: list[dict],
    starting_budget: float,
    capacity_mwh: float,
    efficiency: float,
) -> dict:
    transaction_cost = 1.0
    leg_efficiency = math.sqrt(efficiency)
    cycles = _best_daily_cycles(validation, efficiency)
    actions = {}
    for cycle_index, cycle in enumerate(cycles):
        actions[cycle["buyIndex"]] = ("buy", cycle_index)
        actions[cycle["sellIndex"]] = ("sell", cycle_index)

    cash = starting_budget
    stored_energy = 0.0
    purchase_cost = 0.0
    trades = []
    completed_cycles = []
    equity_curve = []

    for index, point in enumerate(validation):
        action = actions.get(index)
        if action and action[0] == "buy" and stored_energy == 0:
            cost_per_mwh = point["actual"] / leg_efficiency + transaction_cost
            affordable = capacity_mwh
            if cost_per_mwh > 0:
                affordable = min(capacity_mwh, cash / cost_per_mwh)
            stored_energy = max(0.0, affordable)
            purchase_cost = stored_energy * cost_per_mwh
            cash -= purchase_cost
            trades.append(
                {
                    "time": point["time"],
                    "action": "BUY",
                    "marketPrice": point["actual"],
                    "predictedPrice": point["predicted"],
                    "energyMWh": stored_energy,
                    "cashAfter": cash,
                }
            )
        elif action and action[0] == "sell" and stored_energy > 0:
            revenue = stored_energy * (
                point["actual"] * leg_efficiency - transaction_cost
            )
            cash += revenue
            cycle_profit = revenue - purchase_cost
            completed_cycles.append(cycle_profit)
            trades.append(
                {
                    "time": point["time"],
                    "action": "SELL",
                    "marketPrice": point["actual"],
                    "predictedPrice": point["predicted"],
                    "energyMWh": stored_energy,
                    "cashAfter": cash,
                    "cycleProfit": cycle_profit,
                }
            )
            stored_energy = 0.0
            purchase_cost = 0.0

        marked_value = stored_energy * point["actual"] * leg_efficiency
        equity_curve.append({"time": point["time"], "equity": cash + marked_value})

    equity_values = [point["equity"] for point in equity_curve]
    peak = equity_values[0] if equity_values else starting_budget
    max_drawdown = 0.0
    for equity in equity_values:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)

    profit = cash - starting_budget
    return {
        "startingBudget": starting_budget,
        "finalCapital": cash,
        "profit": profit,
        "returnPercent": profit / starting_budget * 100,
        "capacityMWh": capacity_mwh,
        "roundTripEfficiency": efficiency,
        "transactionCostPerMWh": transaction_cost,
        "completedCycles": len(completed_cycles),
        "winningCycles": sum(value > 0 for value in completed_cycles),
        "maxDrawdownPercent": max_drawdown,
        "trades": trades,
        "equityCurve": equity_curve,
    }


def _future_policy_plan(
    forecast: list[dict],
    capacity_mwh: float,
    efficiency: float,
) -> list[dict]:
    cycles = _best_daily_cycles(forecast, efficiency)
    return [
        {
            "buyTime": forecast[cycle["buyIndex"]]["time"],
            "sellTime": forecast[cycle["sellIndex"]]["time"],
            "predictedBuyPrice": forecast[cycle["buyIndex"]]["predicted"],
            "predictedSellPrice": forecast[cycle["sellIndex"]]["predicted"],
            "expectedProfit": cycle["expectedMargin"] * capacity_mwh,
        }
        for cycle in cycles
    ]


def train_electricity_forecast(
    lookback_days: int,
    iterations: int,
    forecast_hours: int,
    starting_budget: float = 1000.0,
    capacity_mwh: float = 1.0,
    efficiency: float = 0.90,
) -> dict:
    if not _training_lock.acquire(blocking=False):
        raise RuntimeError("An electricity-price model is already training.")

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        now = datetime.now(timezone.utc)
        historical_end = datetime.combine(
            now.date() - timedelta(days=1),
            datetime.max.time(),
            tzinfo=timezone.utc,
        ).replace(minute=0, second=0, microsecond=0)
        historical_start = historical_end - timedelta(days=lookback_days + 8)

        prices = _fetch_smard_prices(lookback_days)
        historical_weather = _fetch_historical_weather(
            historical_start,
            historical_end,
        )

        hour_ms = 60 * 60 * 1000
        cutoff_ms = int(historical_end.timestamp() * 1000)
        earliest_sample_ms = int(
            (historical_end - timedelta(days=lookback_days)).timestamp() * 1000
        )

        samples: list[tuple[int, list[float], float]] = []
        for timestamp in sorted(prices):
            if timestamp < earliest_sample_ms or timestamp > cutoff_ms:
                continue
            if (
                timestamp not in historical_weather
                or timestamp - 24 * hour_ms not in prices
                or timestamp - 168 * hour_ms not in prices
            ):
                continue
            features = _feature_vector(
                timestamp,
                historical_weather[timestamp],
                prices[timestamp - 24 * hour_ms],
                prices[timestamp - 168 * hour_ms],
            )
            samples.append((timestamp, features, prices[timestamp]))

        if len(samples) < 240:
            raise RuntimeError(
                "Not enough aligned SMARD and weather observations were available."
            )

        validation_size = min(168, max(48, len(samples) // 5))
        training_samples = samples[:-validation_size]
        validation_samples = samples[-validation_size:]

        x_train_np = np.asarray(
            [sample[1] for sample in training_samples], dtype=np.float32
        )
        y_train_np = np.asarray(
            [sample[2] for sample in training_samples], dtype=np.float32
        ).reshape(-1, 1)
        x_validation_np = np.asarray(
            [sample[1] for sample in validation_samples], dtype=np.float32
        )
        y_validation_np = np.asarray(
            [sample[2] for sample in validation_samples], dtype=np.float32
        ).reshape(-1, 1)

        x_mean = x_train_np.mean(axis=0)
        x_scale = x_train_np.std(axis=0)
        x_scale[x_scale < 1e-6] = 1.0
        y_mean = float(y_train_np.mean())
        y_scale = float(y_train_np.std()) or 1.0

        x_train = torch.tensor((x_train_np - x_mean) / x_scale)
        y_train = torch.tensor((y_train_np - y_mean) / y_scale)
        x_validation = torch.tensor((x_validation_np - x_mean) / x_scale)

        model = ElectricityPriceNetwork(x_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
        loss_function = nn.SmoothL1Loss()
        loss_history = []
        report_every = max(1, iterations // 120)

        model.train()
        for iteration in range(1, iterations + 1):
            optimizer.zero_grad()
            loss = loss_function(model(x_train), y_train)
            loss.backward()
            optimizer.step()
            if iteration == 1 or iteration % report_every == 0 or iteration == iterations:
                loss_history.append(
                    {"iteration": iteration, "loss": float(loss.detach().item())}
                )

        model.eval()
        with torch.no_grad():
            validation_prediction = (
                model(x_validation).numpy().reshape(-1) * y_scale + y_mean
            )

        validation_actual = y_validation_np.reshape(-1)
        validation_metrics = _metrics(validation_actual, validation_prediction)
        baseline_prediction = np.asarray(
            [prices[sample[0] - 24 * hour_ms] for sample in validation_samples]
        )
        baseline_metrics = _metrics(validation_actual, baseline_prediction)

        validation = [
            {
                "time": _ms_to_iso(sample[0]),
                "actual": float(sample[2]),
                "predicted": float(validation_prediction[index]),
                "baseline": float(baseline_prediction[index]),
            }
            for index, sample in enumerate(validation_samples)
        ]
        trading_simulation = _simulate_battery_policy(
            validation,
            starting_budget,
            capacity_mwh,
            efficiency,
        )

        forecast_weather = _fetch_forecast_weather(
            max(3, math.ceil(forecast_hours / 24) + 1)
        )
        forecast_start = max(
            int(now.replace(minute=0, second=0, microsecond=0).timestamp() * 1000),
            cutoff_ms + hour_ms,
        )
        predicted_prices: dict[int, float] = {}
        forecast = []

        for offset in range(forecast_hours):
            timestamp = forecast_start + offset * hour_ms
            weather = forecast_weather.get(timestamp)
            if weather is None:
                continue
            lag_24_timestamp = timestamp - 24 * hour_ms
            lag_168_timestamp = timestamp - 168 * hour_ms
            lag_24 = predicted_prices.get(lag_24_timestamp, prices.get(lag_24_timestamp))
            lag_168 = prices.get(lag_168_timestamp)
            if lag_24 is None or lag_168 is None:
                continue
            features = np.asarray(
                [_feature_vector(timestamp, weather, lag_24, lag_168)],
                dtype=np.float32,
            )
            normalised = torch.tensor((features - x_mean) / x_scale)
            with torch.no_grad():
                predicted = float(model(normalised).item() * y_scale + y_mean)
            predicted_prices[timestamp] = predicted
            forecast.append(
                {
                    "time": _ms_to_iso(timestamp),
                    "predicted": predicted,
                    "actual": prices.get(timestamp),
                    "temperature": weather["temperature_2m"],
                    "windSpeed": weather["wind_speed_100m"],
                    "solarRadiation": weather["shortwave_radiation"],
                    "cloudCover": weather["cloud_cover"],
                }
            )

        global _latest_model
        _latest_model = {
            "model": model,
            "x_mean": x_mean,
            "x_scale": x_scale,
            "y_mean": y_mean,
            "y_scale": y_scale,
            "trained_at": now.isoformat(),
        }

        return {
            "trainedAt": now.isoformat(),
            "trainingSamples": len(training_samples),
            "validationSamples": len(validation_samples),
            "iterations": iterations,
            "lookbackDays": lookback_days,
            "forecastHours": len(forecast),
            "metrics": validation_metrics,
            "baselineMetrics": baseline_metrics,
            "validation": validation,
            "forecast": forecast,
            "lossHistory": loss_history,
            "trading": trading_simulation,
            "plannedTrades": _future_policy_plan(
                forecast,
                capacity_mwh,
                efficiency,
            ),
            "locations": list(GERMANY_LOCATIONS.keys()),
            "sources": [
                "Bundesnetzagentur | SMARD.de",
                "Open-Meteo weather APIs",
            ],
        }
    finally:
        _training_lock.release()
