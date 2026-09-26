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
PREVIOUS_RUNS_WEATHER_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
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

STORAGE_BENCHMARK_2026 = {
    "name": "German utility-scale BESS benchmark",
    "vintage": 2026,
    "capexPerKW": 400.0,
    "gridConnectionPerKW": 100.0,
    "opexPerKWhYear": 12.0,
    "lifetimeYears": 15,
    "availability": 0.95,
    "financingRate": 0.08,
    "revenueShare": 0.06,
    "sourceUrl": "https://www.bundesnetzagentur.de/DE/Beschlusskammern/GBK/GBK_Termine/Downloads/2026/01_2026/30_01/4_BVES_ECO.pdf?__blob=publicationFile&v=2",
}


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


def _fetch_day_ahead_weather(
    start: datetime,
    end: datetime,
) -> dict[int, dict[str, float]]:
    """Weather values that were available 24 hours before each delivery hour."""
    variables = [
        f"{name}_previous_day1" for name in WEATHER_VARIABLES.split(",")
    ]
    params = _weather_params(start.date().isoformat(), end.date().isoformat())
    params["hourly"] = ",".join(variables)
    payload = _fetch_json(PREVIOUS_RUNS_WEATHER_URL, params)
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
            for base_name, api_name in zip(bucket, variables):
                value = hourly[api_name][index]
                if value is not None:
                    bucket[base_name].append(float(value))
    return {
        timestamp: {
            name: float(np.mean(values)) if values else 0.0
            for name, values in weather.items()
        }
        for timestamp, weather in combined.items()
        if all(weather[name] for name in weather)
    }


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


def _calibrated_storage_rent(capacity_mwh: float, power_mw: float) -> tuple[float, dict]:
    benchmark = dict(STORAGE_BENCHMARK_2026)
    rate = benchmark["financingRate"]
    years = benchmark["lifetimeYears"]
    capital_recovery_factor = rate * (1 + rate) ** years / ((1 + rate) ** years - 1)
    annual_capital_cost = (
        power_mw
        * 1000
        * (benchmark["capexPerKW"] + benchmark["gridConnectionPerKW"])
        * capital_recovery_factor
    )
    annual_opex = capacity_mwh * 1000 * benchmark["opexPerKWhYear"]
    annual_cost = (annual_capital_cost + annual_opex) / benchmark["availability"]
    rent = annual_cost / 365 / capacity_mwh
    benchmark.update(
        {
            "capitalRecoveryFactor": capital_recovery_factor,
            "annualizedCost": annual_cost,
            "calculatedRentPerMWhDay": rent,
        }
    )
    return rent, benchmark


def _fit_model(
    samples: list[tuple[int, list[float], float]],
    iterations: int,
    seed: int,
) -> tuple[ElectricityPriceNetwork, np.ndarray, np.ndarray, float, float, list[dict]]:
    torch.manual_seed(seed)
    x_values = np.asarray([sample[1] for sample in samples], dtype=np.float32)
    y_values = np.asarray(
        [sample[2] for sample in samples], dtype=np.float32
    ).reshape(-1, 1)
    x_mean = x_values.mean(axis=0)
    x_scale = x_values.std(axis=0)
    x_scale[x_scale < 1e-6] = 1.0
    y_mean = float(y_values.mean())
    y_scale = float(y_values.std()) or 1.0
    x_train = torch.tensor((x_values - x_mean) / x_scale)
    y_train = torch.tensor((y_values - y_mean) / y_scale)
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
    return model, x_mean, x_scale, y_mean, y_scale, loss_history


def _predict_samples(
    model: ElectricityPriceNetwork,
    samples: list[tuple[int, list[float], float]],
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    y_mean: float,
    y_scale: float,
) -> np.ndarray:
    values = np.asarray([sample[1] for sample in samples], dtype=np.float32)
    with torch.no_grad():
        return (
            model(torch.tensor((values - x_mean) / x_scale))
            .numpy()
            .reshape(-1)
            * y_scale
            + y_mean
        )


def _best_daily_cycles(
    points: list[dict],
    efficiency: float,
    price_key: str,
    variable_cost_per_mwh: float,
) -> list[dict]:
    leg_efficiency = math.sqrt(efficiency)
    cycles = []
    for start in range(0, len(points), 24):
        day = points[start : start + 24]
        best = None
        for buy_index in range(len(day) - 1):
            for sell_index in range(buy_index + 1, len(day)):
                expected_margin = (
                    day[sell_index][price_key] * leg_efficiency
                    - day[buy_index][price_key] / leg_efficiency
                    - 2 * variable_cost_per_mwh
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
    power_mw: float,
    battery_cost_per_kwh: float,
    market_fee_per_mwh: float,
    degradation_cost_per_mwh: float,
    self_discharge_percent_per_day: float,
    storage_model: str,
    rental_cost_per_mwh_day: float,
    operator_revenue_share: float,
    price_key: str = "predicted",
    name: str = "Model policy",
) -> dict:
    variable_cost = market_fee_per_mwh + degradation_cost_per_mwh / 2
    leg_efficiency = math.sqrt(efficiency)
    cycles = _best_daily_cycles(validation, efficiency, price_key, variable_cost)
    actions = {}
    for cycle_index, cycle in enumerate(cycles):
        actions[cycle["buyIndex"]] = ("buy", cycle_index)
        actions[cycle["sellIndex"]] = ("sell", cycle_index)

    simulation_days = len(validation) / 24
    battery_value = (
        capacity_mwh * 1000 * battery_cost_per_kwh
        if storage_model == "owned"
        else 0.0
    )
    rental_cost = (
        capacity_mwh * rental_cost_per_mwh_day * simulation_days
        if storage_model == "rented"
        else 0.0
    )
    cash = starting_budget - rental_cost
    invested_capital = starting_budget + battery_value
    stored_energy = 0.0
    purchase_cost = 0.0
    trades = []
    completed_cycles = []
    revenue_share_paid = 0.0
    equity_curve = []
    if validation:
        first_timestamp = _iso_to_ms(validation[0]["time"])
        equity_curve.append(
            {
                "time": _ms_to_iso(first_timestamp - 60 * 60 * 1000),
                "equity": starting_budget + battery_value,
                "event": "Starting capital",
            }
        )
        if rental_cost > 0:
            equity_curve.append(
                {
                    "time": _ms_to_iso(first_timestamp - 1000),
                    "equity": cash + battery_value,
                    "event": "Storage rent paid",
                }
            )

    for index, point in enumerate(validation):
        action = actions.get(index)
        if action and action[0] == "buy" and stored_energy == 0:
            cost_per_mwh = point["actual"] / leg_efficiency + variable_cost
            affordable = min(capacity_mwh, power_mw)
            if cost_per_mwh > 0:
                affordable = min(capacity_mwh, power_mw, cash / cost_per_mwh)
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
            cycle = cycles[action[1]]
            elapsed_hours = max(1, cycle["sellIndex"] - cycle["buyIndex"])
            retained_energy = stored_energy * (
                1 - self_discharge_percent_per_day / 100
            ) ** (elapsed_hours / 24)
            revenue = retained_energy * (
                point["actual"] * leg_efficiency - variable_cost
            )
            gross_cycle_profit = revenue - purchase_cost
            revenue_share = (
                max(0.0, gross_cycle_profit) * operator_revenue_share
                if storage_model == "rented"
                else 0.0
            )
            revenue_share_paid += revenue_share
            cash += revenue - revenue_share
            cycle_profit = gross_cycle_profit - revenue_share
            completed_cycles.append(cycle_profit)
            trades.append(
                {
                    "time": point["time"],
                    "action": "SELL",
                    "marketPrice": point["actual"],
                    "predictedPrice": point["predicted"],
                    "energyMWh": retained_energy,
                    "cashAfter": cash,
                    "cycleProfit": cycle_profit,
                }
            )
            stored_energy = 0.0
            purchase_cost = 0.0

        marked_value = stored_energy * point["actual"] * leg_efficiency
        equity_curve.append(
            {"time": point["time"], "equity": cash + battery_value + marked_value}
        )

    equity_values = [point["equity"] for point in equity_curve]
    peak = equity_values[0] if equity_values else invested_capital
    max_drawdown = 0.0
    for equity in equity_values:
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - equity) / peak * 100)

    profit = cash - starting_budget
    trading_profit_before_storage_costs = profit + rental_cost + revenue_share_paid
    return {
        "name": name,
        "storageModel": storage_model,
        "startingBudget": starting_budget,
        "batteryInvestment": battery_value,
        "storageRentalCost": rental_cost,
        "rentalCostPerMWhDay": rental_cost_per_mwh_day,
        "operatorRevenueSharePercent": operator_revenue_share * 100,
        "revenueSharePaid": revenue_share_paid,
        "tradingProfitBeforeStorageCosts": trading_profit_before_storage_costs,
        "investedCapital": invested_capital,
        "finalCapital": invested_capital + profit,
        "finalCash": cash,
        "profit": profit,
        "returnPercent": profit / invested_capital * 100,
        "capacityMWh": capacity_mwh,
        "powerMW": power_mw,
        "roundTripEfficiency": efficiency,
        "marketFeePerMWh": market_fee_per_mwh,
        "degradationCostPerMWh": degradation_cost_per_mwh,
        "selfDischargePercentPerDay": self_discharge_percent_per_day,
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
    power_mw: float,
    market_fee_per_mwh: float,
    degradation_cost_per_mwh: float,
    storage_model: str,
    operator_revenue_share: float,
) -> list[dict]:
    variable_cost = market_fee_per_mwh + degradation_cost_per_mwh / 2
    cycles = _best_daily_cycles(
        forecast, efficiency, "predicted", variable_cost
    )
    return [
        {
            "buyTime": forecast[cycle["buyIndex"]]["time"],
            "sellTime": forecast[cycle["sellIndex"]]["time"],
            "predictedBuyPrice": forecast[cycle["buyIndex"]]["predicted"],
            "predictedSellPrice": forecast[cycle["sellIndex"]]["predicted"],
            "expectedProfit": (
                cycle["expectedMargin"]
                * min(capacity_mwh, power_mw)
                * (1 - operator_revenue_share if storage_model == "rented" else 1)
            ),
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
    power_mw: float = 0.5,
    battery_cost_per_kwh: float = 400.0,
    market_fee_per_mwh: float = 3.0,
    degradation_cost_per_mwh: float = 20.0,
    self_discharge_percent_per_day: float = 0.2,
    storage_model: str = "owned",
    rental_cost_per_mwh_day: float = 150.0,
    operator_revenue_share: float = 0.10,
    use_benchmark_storage_costs: bool = True,
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
        historical_weather = _fetch_day_ahead_weather(
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

        # Expanding-window walk-forward validation. For every 24-hour block the
        # model is rebuilt using only observations strictly before that block.
        validation_predictions: list[float] = []
        for offset in range(0, len(validation_samples), 24):
            fold_training = training_samples + validation_samples[:offset]
            fold_validation = validation_samples[offset : offset + 24]
            fold_model, fold_x_mean, fold_x_scale, fold_y_mean, fold_y_scale, _ = (
                _fit_model(fold_training, iterations, 42 + offset)
            )
            validation_predictions.extend(
                _predict_samples(
                    fold_model,
                    fold_validation,
                    fold_x_mean,
                    fold_x_scale,
                    fold_y_mean,
                    fold_y_scale,
                ).tolist()
            )

        validation_prediction = np.asarray(validation_predictions)
        validation_actual = np.asarray(
            [sample[2] for sample in validation_samples], dtype=np.float32
        )
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
        storage_benchmark = None
        if storage_model == "rented" and use_benchmark_storage_costs:
            rental_cost_per_mwh_day, storage_benchmark = _calibrated_storage_rent(
                capacity_mwh, power_mw
            )
            operator_revenue_share = STORAGE_BENCHMARK_2026["revenueShare"]

        simulation_args = (
            validation,
            starting_budget,
            capacity_mwh,
            efficiency,
            power_mw,
            battery_cost_per_kwh,
            market_fee_per_mwh,
            degradation_cost_per_mwh,
            self_discharge_percent_per_day,
            storage_model,
            rental_cost_per_mwh_day,
            operator_revenue_share,
        )
        trading_simulation = _simulate_battery_policy(*simulation_args)
        comparison_policies = [
            trading_simulation,
            _simulate_battery_policy(
                *simulation_args,
                price_key="baseline",
                name="24 h naive policy",
            ),
            _simulate_battery_policy(
                *simulation_args,
                price_key="actual",
                name="Perfect foresight ceiling",
            ),
        ]

        # The production model can use all known observations because it only
        # predicts hours after the historical cutoff.
        model, x_mean, x_scale, y_mean, y_scale, loss_history = _fit_model(
            samples, iterations, 2026
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
            "policyComparison": comparison_policies,
            "validationMethod": "Daily expanding-window walk-forward",
            "weatherDataMethod": "Archived weather forecast issued 24 hours earlier",
            "storageBenchmark": storage_benchmark,
            "plannedTrades": _future_policy_plan(
                forecast,
                capacity_mwh,
                efficiency,
                power_mw,
                market_fee_per_mwh,
                degradation_cost_per_mwh,
                storage_model,
                operator_revenue_share,
            ),
            "locations": list(GERMANY_LOCATIONS.keys()),
            "sources": [
                "Bundesnetzagentur | SMARD.de",
                "Open-Meteo weather APIs",
            ],
        }
    finally:
        _training_lock.release()
