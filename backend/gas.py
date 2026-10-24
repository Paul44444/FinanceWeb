from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yfinance as yf

from backend.electricity import (
    GERMANY_LOCATIONS,
    _fetch_json,
    _fit_model,
    _metrics,
    _predict_samples,
)


GAS_TICKER = "TTF=F"
EXTERNAL_TICKERS = {
    "henryHub": "NG=F",
    "brent": "BZ=F",
    "eurUsd": "EURUSD=X",
}
PREVIOUS_RUNS_WEATHER_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
# Gas demand is driven by the whole north-west European heating region.
GAS_LOCATIONS = {
    **GERMANY_LOCATIONS,
    "Amsterdam": (52.37, 4.90),
    "Paris": (48.86, 2.35),
}
# Archived forecasts are complete for temperature and wind from this date on.
WEATHER_ARCHIVE_START = date(2024, 4, 1)
WEATHER_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "gas_weather"
FORECAST_HORIZON_DAYS = 5
# Eurostat heating degree days: 18 °C minus the daily mean when it is <= 15 °C.
HDD_BASE = 18.0
HDD_THRESHOLD = 15.0
# ESMA retail limit for commodity CFDs other than gold (10:1).
CFD_MINIMUM_MARGIN = 0.10
TRADING_DAYS_PER_YEAR = 252
# Daily gas returns are mostly noise; L2 regularisation keeps the network from
# memorising it and producing overconfident signals.
WEIGHT_DECAY = 0.05

_training_lock = Lock()
_weather_cache: dict[tuple[str, str], tuple[float, dict]] = {}


def _hdd(temperature: float) -> float:
    return HDD_BASE - temperature if temperature <= HDD_THRESHOLD else 0.0


def _fetch_closes(ticker: str, start: date) -> dict[date, float]:
    history = yf.Ticker(ticker).history(
        start=start.isoformat(), interval="1d", auto_adjust=False
    )
    return {
        index.date(): float(close)
        for index, close in history["Close"].items()
        if close is not None and not math.isnan(close)
    }


def _drop_unsettled_session(closes: dict[date, float]) -> dict[date, float]:
    """Remove today's bar while ICE Endex is still trading (settles ~17:30 CET)."""
    now = datetime.now(ZoneInfo("Europe/Amsterdam"))
    if now.hour * 60 + now.minute < 18 * 60 + 15:
        closes.pop(now.date(), None)
    return closes


def _front_month_expiry(year: int, month: int) -> date:
    """TTF futures stop trading two business days before the delivery month."""
    day = date(year, month, 1) + timedelta(days=32)
    day = day.replace(day=1) - timedelta(days=1)
    weekdays_seen = 0
    while True:
        if day.weekday() < 5:
            weekdays_seen += 1
            if weekdays_seen == 2:
                return day
        day -= timedelta(days=1)


def _roll_days(trading_days: list[date]) -> set[date]:
    """Days whose continuous front-month return spans two different contracts."""
    rolls = set()
    for previous, current in zip(trading_days, trading_days[1:]):
        expiry = _front_month_expiry(previous.year, previous.month)
        if previous <= expiry < current:
            rolls.add(current)
    return rolls


def _fetch_issued_weather(start: date, end: date) -> dict[date, dict[int, dict]]:
    """Daily HDD and wind per archived forecast lag, averaged over GAS_LOCATIONS.

    Returns weather[target_date][lag] = {"hdd": ..., "wind": ...}, where lag is the
    number of days between issuing the forecast and the target date. Archived
    forecasts never change, so completed months are cached on disk.
    """
    stable_until = datetime.now(timezone.utc).date() - timedelta(
        days=FORECAST_HORIZON_DAYS + 4
    )
    archived_months = []
    month = start.replace(day=1)
    while True:
        month_end = (month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        if month_end > stable_until or month_end > end:
            break
        archived_months.append((max(month, start), month_end))
        month = month_end + timedelta(days=1)

    with ThreadPoolExecutor(max_workers=4) as executor:
        chunks = list(executor.map(lambda span: _cached_month_weather(*span), archived_months))
    weather: dict[date, dict[int, dict]] = {}
    for chunk in chunks:
        weather.update(chunk)
    if month <= end:
        key = (month.isoformat(), end.isoformat())
        cached = _weather_cache.get(key)
        if cached is None or time.time() - cached[0] > 3 * 60 * 60:
            cached = (time.time(), _download_issued_weather(month, end))
            _weather_cache[key] = cached
        weather.update(cached[1])
    return weather


def _cached_month_weather(start: date, end: date) -> dict[date, dict[int, dict]]:
    path = WEATHER_CACHE_DIR / f"{start.isoformat()}_{end.isoformat()}.json"
    if path.exists():
        stored = json.loads(path.read_text())
        return {
            date.fromisoformat(day): {int(lag): values for lag, values in lags.items()}
            for day, lags in stored.items()
        }
    weather = _download_issued_weather(start, end)
    WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({day.isoformat(): lags for day, lags in weather.items()})
    )
    return weather


def _download_issued_weather(start: date, end: date) -> dict[date, dict[int, dict]]:
    lags = range(2, FORECAST_HORIZON_DAYS + 3)
    variables = [
        f"{name}_previous_day{lag}"
        for name in ("temperature_2m", "wind_speed_100m")
        for lag in lags
    ]
    payload = _fetch_json(
        PREVIOUS_RUNS_WEATHER_URL,
        {
            "latitude": ",".join(str(value[0]) for value in GAS_LOCATIONS.values()),
            "longitude": ",".join(str(value[1]) for value in GAS_LOCATIONS.values()),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(variables),
            "timezone": "GMT",
        },
    )
    locations = payload if isinstance(payload, list) else [payload]
    # per_location[location][target_date][variable] -> hourly values
    combined: dict[date, dict[int, dict[str, list[float]]]] = {}
    for location in locations:
        hourly = location["hourly"]
        daily: dict[date, dict[str, list[float]]] = {}
        for index, time_value in enumerate(hourly["time"]):
            day = date.fromisoformat(time_value[:10])
            bucket = daily.setdefault(day, {name: [] for name in variables})
            for name in variables:
                value = hourly[name][index]
                if value is not None:
                    bucket[name].append(float(value))
        for day, values in daily.items():
            for lag in lags:
                temperatures = values[f"temperature_2m_previous_day{lag}"]
                winds = values[f"wind_speed_100m_previous_day{lag}"]
                if len(temperatures) < 20 or len(winds) < 20:
                    continue
                entry = combined.setdefault(day, {}).setdefault(
                    lag, {"hdd": [], "wind": []}
                )
                entry["hdd"].append(_hdd(float(np.mean(temperatures))))
                entry["wind"].append(float(np.mean(winds)))

    location_count = len(locations)
    return {
        day: {
            lag: {"hdd": float(np.mean(values["hdd"])), "wind": float(np.mean(values["wind"]))}
            for lag, values in lag_values.items()
            if len(values["hdd"]) == location_count
        }
        for day, lag_values in combined.items()
    }


def _weather_outlook(
    weather: dict[date, dict[int, dict]], decision_day: date, issue_offset: int
) -> dict[str, float] | None:
    """Mean forecast for the next days, issued `issue_offset` days before decision_day."""
    hdd_values, wind_values = [], []
    for ahead in range(1, FORECAST_HORIZON_DAYS + 1):
        lag = ahead + issue_offset
        values = weather.get(decision_day + timedelta(days=ahead), {}).get(lag)
        if values is None:
            return None
        hdd_values.append(values["hdd"])
        wind_values.append(values["wind"])
    return {"hdd": float(np.mean(hdd_values)), "wind": float(np.mean(wind_values))}


def _fetch_forecast_weather_daily(days: int) -> list[dict]:
    payload = _fetch_json(
        FORECAST_WEATHER_URL,
        {
            "latitude": ",".join(str(value[0]) for value in GAS_LOCATIONS.values()),
            "longitude": ",".join(str(value[1]) for value in GAS_LOCATIONS.values()),
            "daily": "temperature_2m_mean,wind_speed_10m_max",
            "hourly": "wind_speed_100m",
            "forecast_days": days,
            "timezone": "GMT",
        },
    )
    locations = payload if isinstance(payload, list) else [payload]
    by_day: dict[str, dict[str, list[float]]] = {}
    for location in locations:
        daily = location["daily"]
        hourly = location["hourly"]
        hourly_wind: dict[str, list[float]] = {}
        for index, time_value in enumerate(hourly["time"]):
            value = hourly["wind_speed_100m"][index]
            if value is not None:
                hourly_wind.setdefault(time_value[:10], []).append(float(value))
        for index, day in enumerate(daily["time"]):
            temperature = daily["temperature_2m_mean"][index]
            if temperature is None or day not in hourly_wind:
                continue
            bucket = by_day.setdefault(day, {"temperature": [], "hdd": [], "wind": []})
            bucket["temperature"].append(float(temperature))
            bucket["hdd"].append(_hdd(float(temperature)))
            bucket["wind"].append(float(np.mean(hourly_wind[day])))
    return [
        {
            "date": day,
            "temperature": float(np.mean(values["temperature"])),
            "hdd": float(np.mean(values["hdd"])),
            "windSpeed": float(np.mean(values["wind"])),
        }
        for day, values in sorted(by_day.items())
    ]


def _latest_before(series: dict[date, float], days: list[date], day: date, steps: int):
    """Return the `steps` most recent closes strictly before `day` (oldest first)."""
    position = np.searchsorted(np.asarray(days, dtype="datetime64[D]"), np.datetime64(day))
    if position < steps:
        return None
    return [series[days[index]] for index in range(position - steps, position)]


def _feature_vector(
    day: date,
    closes: list[float],
    externals: dict[str, list[float]],
    outlook: dict[str, float],
    previous_outlook: dict[str, float],
) -> list[float]:
    log_closes = np.log(np.asarray(closes))
    returns = np.diff(log_closes)
    weekday_angle = 2 * math.pi * day.weekday() / 7
    year_angle = 2 * math.pi * day.timetuple().tm_yday / 365.25
    return [
        math.sin(weekday_angle),
        math.cos(weekday_angle),
        math.sin(year_angle),
        math.cos(year_angle),
        float(returns[-1]),
        float(log_closes[-1] - log_closes[-6]),
        float(log_closes[-1] - log_closes[-21]),
        float(np.std(returns[-20:])),
        float(log_closes[-1] - np.mean(log_closes[-20:])),
        *(
            float(math.log(values[-1] / values[-2]))
            for values in externals.values()
        ),
        outlook["hdd"],
        outlook["hdd"] - previous_outlook["hdd"],
        outlook["wind"],
        outlook["wind"] - previous_outlook["wind"],
    ]


def _position_limits(
    instrument: str,
    equity: float,
    price: float,
    leverage: float,
    margin_rate: float,
    contract_size_mwh: float,
) -> float:
    """Largest position (MWh) allowed by target leverage and margin."""
    if equity <= 0 or price <= 0:
        return 0.0
    if instrument == "future":
        contract_value = contract_size_mwh * price
        contracts = min(
            math.floor(leverage * equity / contract_value),
            math.floor(equity / (contract_value * margin_rate)),
        )
        return max(0, contracts) * contract_size_mwh
    effective_margin = max(margin_rate, CFD_MINIMUM_MARGIN)
    return min(leverage * equity, equity / effective_margin) / price


def _simulate_gas_policy(
    validation: list[dict],
    instrument: str,
    starting_budget: float,
    leverage: float,
    spread_per_mwh: float,
    commission_per_mwh: float,
    financing_rate: float,
    margin_rate: float,
    contract_size_mwh: float,
    prediction_key: str = "predictedChange",
    name: str = "Model policy",
) -> dict:
    side_cost = spread_per_mwh / 2 + commission_per_mwh
    equity = starting_budget
    position = 0.0
    transaction_costs = 0.0
    financing_costs = 0.0
    blocked_signals = 0
    trades = []
    daily_pnl = []
    active_days = 0
    winning_days = 0
    correct_direction = 0
    stopped = False
    equity_curve = []
    if validation:
        equity_curve.append(
            {"time": validation[0]["date"], "equity": equity, "event": "Starting capital"}
        )

    for point in validation:
        price = point["close"]
        holding_days = (
            date.fromisoformat(point["nextDate"]) - date.fromisoformat(point["date"])
        ).days
        financing_per_mwh = (
            price * financing_rate * holding_days / 365 if instrument == "cfd" else 0.0
        )
        expected_change = point.get(prediction_key)
        desired_side = 0
        if point["tradable"] and expected_change is not None and not stopped:
            current_side = int(np.sign(position))
            for side in (1, -1):
                required_edge = financing_per_mwh + (
                    0.0 if side == current_side else 2 * side_cost
                )
                if side * expected_change > required_edge:
                    desired_side = side

        if desired_side == int(np.sign(position)) and position != 0:
            target = position
        elif desired_side == 0:
            target = 0.0
        else:
            target = desired_side * _position_limits(
                instrument, equity, price, leverage, margin_rate, contract_size_mwh
            )
            if target == 0:
                blocked_signals += 1

        change = target - position
        if change != 0:
            cost = abs(change) * side_cost
            transaction_costs += cost
            equity -= cost
            trades.append(
                {
                    "time": point["date"],
                    "action": "BUY" if change > 0 else "SELL",
                    "positionAfterMWh": target,
                    "quantityMWh": abs(change),
                    "marketPrice": price,
                    "predictedChange": expected_change,
                    "equityAfter": equity,
                }
            )
        position = target

        financing = abs(position) * financing_per_mwh
        pnl = position * (point["nextClose"] - price) - financing
        financing_costs += financing
        equity += pnl
        if position != 0:
            active_days += 1
            winning_days += pnl > 0
            correct_direction += np.sign(position) == np.sign(
                point["nextClose"] - price
            )
            daily_pnl.append(pnl)
        if equity <= 0:
            # Negative-balance protection for CFDs; futures losses stay with the trader.
            equity = 0.0 if instrument == "cfd" else equity
            stopped = True
        equity_curve.append({"time": point["nextDate"], "equity": equity})

    if position != 0 and validation:
        cost = abs(position) * side_cost
        transaction_costs += cost
        equity -= cost
        trades.append(
            {
                "time": validation[-1]["nextDate"],
                "action": "SELL" if position > 0 else "BUY",
                "positionAfterMWh": 0.0,
                "quantityMWh": abs(position),
                "marketPrice": validation[-1]["nextClose"],
                "predictedChange": None,
                "equityAfter": equity,
            }
        )
        equity_curve[-1]["equity"] = equity

    equity_values = np.asarray([point["equity"] for point in equity_curve])
    peak = equity_values[0] if len(equity_values) else starting_budget
    max_drawdown = 0.0
    for value in equity_values:
        peak = max(peak, value)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - value) / peak * 100)

    sharpe = 0.0
    if len(equity_values) > 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(equity_values) / equity_values[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(
                np.mean(returns) / np.std(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)
            )

    mean_daily_pnl = float(np.mean(daily_pnl)) if daily_pnl else 0.0
    confidence = 0.0
    if len(daily_pnl) > 1:
        confidence = float(1.96 * np.std(daily_pnl, ddof=1) / math.sqrt(len(daily_pnl)))
    profit = equity - starting_budget
    return {
        "name": name,
        "instrument": instrument,
        "startingBudget": starting_budget,
        "finalCapital": equity,
        "profit": profit,
        "returnPercent": profit / starting_budget * 100,
        "tradingProfitBeforeCosts": profit + transaction_costs + financing_costs,
        "transactionCosts": transaction_costs,
        "financingCosts": financing_costs,
        "leverage": leverage,
        "spreadPerMWh": spread_per_mwh,
        "commissionPerMWh": commission_per_mwh,
        "financingRatePercent": financing_rate * 100,
        "marginPercent": (
            max(margin_rate, CFD_MINIMUM_MARGIN) if instrument == "cfd" else margin_rate
        )
        * 100,
        "contractSizeMWh": contract_size_mwh,
        "tradeCount": len(trades),
        "activeDays": active_days,
        "winningDays": int(winning_days),
        "directionHitRatePercent": (
            correct_direction / active_days * 100 if active_days else 0.0
        ),
        "blockedSignals": blocked_signals,
        "meanDailyPnl": mean_daily_pnl,
        "meanDailyPnlConfidence": confidence,
        "sharpeRatio": sharpe,
        "maxDrawdownPercent": max_drawdown,
        "stoppedOut": stopped,
        "trades": trades,
        "equityCurve": equity_curve,
    }


def _binomial_p_value(successes: int, trials: int) -> float:
    """Two-sided p-value for a hit rate against a fair coin (normal approximation)."""
    if trials == 0:
        return 1.0
    z = (successes - trials / 2) / math.sqrt(trials / 4)
    return float(math.erfc(abs(z) / math.sqrt(2)))


def train_gas_forecast(
    lookback_days: int,
    iterations: int,
    test_days: int,
    starting_budget: float = 1000.0,
    instrument: str = "cfd",
    leverage: float = 1.0,
    spread_per_mwh: float = 0.20,
    commission_per_mwh: float = 0.0,
    financing_rate: float = 0.06,
    margin_rate: float = 0.10,
    contract_size_mwh: float = 730.0,
    progress_callback=None,
) -> dict:
    def report(stage: str, step: int = 0, total: int = 0) -> None:
        if progress_callback:
            progress_callback({"stage": stage, "step": step, "total": total})

    if not _training_lock.acquire(blocking=False):
        raise RuntimeError("A gas-price model is already training.")

    try:
        torch.manual_seed(42)
        np.random.seed(42)
        now = datetime.now(timezone.utc)
        history_start = max(
            WEATHER_ARCHIVE_START - timedelta(days=45),
            now.date() - timedelta(days=lookback_days + 45),
        )

        report("Downloading TTF, Henry Hub, Brent and EUR/USD prices")
        gas = _drop_unsettled_session(_fetch_closes(GAS_TICKER, history_start))
        trading_days = sorted(gas)
        if len(trading_days) < 60:
            raise RuntimeError("Not enough TTF price history was available.")
        externals = {
            name: _fetch_closes(ticker, history_start - timedelta(days=10))
            for name, ticker in EXTERNAL_TICKERS.items()
        }
        external_days = {name: sorted(series) for name, series in externals.items()}
        roll_days = _roll_days(trading_days)
        last_day = trading_days[-1]
        report("Loading archived weather forecasts")
        weather = _fetch_issued_weather(
            max(WEATHER_ARCHIVE_START, history_start),
            last_day + timedelta(days=FORECAST_HORIZON_DAYS + 1),
        )

        def features_for(index: int) -> list[float] | None:
            if index < 21:
                return None
            day = trading_days[index]
            outlook = _weather_outlook(weather, day, 1)
            previous_outlook = _weather_outlook(weather, day, 2)
            if outlook is None or previous_outlook is None:
                return None
            external_values = {}
            for name, series in externals.items():
                values = _latest_before(series, external_days[name], day, 2)
                if values is None:
                    return None
                external_values[name] = values
            closes = [gas[trading_days[offset]] for offset in range(index - 20, index + 1)]
            return _feature_vector(day, closes, external_values, outlook, previous_outlook)

        earliest_decision = now.date() - timedelta(days=lookback_days)
        decisions = []
        for index in range(len(trading_days) - 1):
            day = trading_days[index]
            if day < earliest_decision:
                continue
            next_day = trading_days[index + 1]
            features = features_for(index)
            decisions.append(
                {
                    "index": index,
                    "date": day,
                    "nextDate": next_day,
                    "features": features,
                    "tradable": features is not None and next_day not in roll_days,
                    "target": math.log(gas[next_day] / gas[day]),
                }
            )

        usable = [entry for entry in decisions if entry["tradable"]]
        if len(usable) < test_days + 120:
            raise RuntimeError(
                f"Only {len(usable)} leakage-safe trading days are available; "
                f"reduce the walk-forward test period or extend the history."
            )

        test_start = usable[-test_days]["date"]
        training_pool = [
            (entry["index"], entry["features"], entry["target"]) for entry in usable
        ]
        test_entries = [entry for entry in decisions if entry["date"] >= test_start]

        # Daily expanding-window walk-forward: the model for decision day t is
        # trained only on samples whose outcome was known at t (target day <= t).
        validation = []
        for step, entry in enumerate(test_entries, start=1):
            report("Walk-forward retraining", step, len(test_entries))
            predicted_return = None
            if entry["tradable"]:
                fold = [sample for sample in training_pool if sample[0] < entry["index"]]
                model, x_mean, x_scale, y_mean, y_scale, _ = _fit_model(
                    fold, iterations, 42 + entry["index"], WEIGHT_DECAY
                )
                predicted_return = float(
                    _predict_samples(
                        model,
                        [(entry["index"], entry["features"], 0.0)],
                        x_mean,
                        x_scale,
                        y_mean,
                        y_scale,
                    )[0]
                )
            close = gas[entry["date"]]
            next_close = gas[entry["nextDate"]]
            previous_close = gas[trading_days[entry["index"] - 1]]
            validation.append(
                {
                    "date": entry["date"].isoformat(),
                    "nextDate": entry["nextDate"].isoformat(),
                    "close": close,
                    "nextClose": next_close,
                    "tradable": entry["tradable"],
                    "rollDay": entry["nextDate"] in roll_days,
                    "predictedReturn": predicted_return,
                    "predicted": (
                        close * math.exp(predicted_return)
                        if predicted_return is not None
                        else None
                    ),
                    "predictedChange": (
                        close * math.expm1(predicted_return)
                        if predicted_return is not None
                        else None
                    ),
                    "actualChange": next_close - close,
                    "momentumChange": close - previous_close,
                    # Large enough to clear any cost threshold, finite for JSON.
                    "alwaysLongChange": 1e6,
                }
            )

        scored = [point for point in validation if point["predicted"] is not None]
        actual_prices = np.asarray([point["nextClose"] for point in scored])
        model_prices = np.asarray([point["predicted"] for point in scored])
        random_walk_prices = np.asarray([point["close"] for point in scored])
        actual_returns = np.asarray(
            [math.log(point["nextClose"] / point["close"]) for point in scored]
        )
        predicted_returns = np.asarray([point["predictedReturn"] for point in scored])
        direction_hits = int(
            np.sum(np.sign(predicted_returns) == np.sign(actual_returns))
        )
        return_correlation = 0.0
        if np.std(predicted_returns) > 0 and np.std(actual_returns) > 0:
            return_correlation = float(np.corrcoef(predicted_returns, actual_returns)[0, 1])

        simulation_args = (
            instrument,
            starting_budget,
            leverage,
            spread_per_mwh,
            commission_per_mwh,
            financing_rate,
            margin_rate,
            contract_size_mwh,
        )
        trading_simulation = _simulate_gas_policy(validation, *simulation_args)
        comparison_policies = [
            trading_simulation,
            _simulate_gas_policy(
                validation,
                *simulation_args,
                prediction_key="momentumChange",
                name="Momentum (yesterday's move)",
            ),
            _simulate_gas_policy(
                validation,
                *simulation_args,
                prediction_key="alwaysLongChange",
                name="Buy and hold",
            ),
            _simulate_gas_policy(
                validation,
                *simulation_args,
                prediction_key="actualChange",
                name="Perfect foresight ceiling",
            ),
        ]

        report("Training the production model")
        # The production model may use every known outcome because it only
        # predicts the move after the latest settled close.
        model, x_mean, x_scale, y_mean, y_scale, loss_history = _fit_model(
            training_pool, iterations, 2026, WEIGHT_DECAY
        )
        signal = None
        latest_features = features_for(len(trading_days) - 1)
        if latest_features is not None:
            predicted_return = float(
                _predict_samples(
                    model,
                    [(0, latest_features, 0.0)],
                    x_mean,
                    x_scale,
                    y_mean,
                    y_scale,
                )[0]
            )
            close = gas[last_day]
            predicted_change = close * math.expm1(predicted_return)
            financing_per_mwh = (
                close * financing_rate / 365 if instrument == "cfd" else 0.0
            )
            required_edge = financing_per_mwh + spread_per_mwh + 2 * commission_per_mwh
            position_mwh = _position_limits(
                instrument, starting_budget, close, leverage, margin_rate, contract_size_mwh
            )
            action = "Stay flat"
            if abs(predicted_change) > required_edge:
                action = "Long" if predicted_change > 0 else "Short"
            signal = {
                "decisionDate": last_day.isoformat(),
                "lastClose": close,
                "predictedReturnPercent": math.expm1(predicted_return) * 100,
                "predictedClose": close + predicted_change,
                "predictedChange": predicted_change,
                "requiredEdge": required_edge,
                "action": action,
                "positionMWh": position_mwh if action != "Stay flat" else 0.0,
                "nextIsRollDay": False,
            }
            following = last_day + timedelta(days=1)
            while following.weekday() >= 5:
                following += timedelta(days=1)
            if following in _roll_days([last_day, following]):
                signal["nextIsRollDay"] = True
                signal["action"] = "Stay flat"
                signal["positionMWh"] = 0.0

        history = [
            {"date": day.isoformat(), "close": gas[day], "rollDay": day in roll_days}
            for day in trading_days
            if day >= earliest_decision
        ]

        return {
            "trainedAt": now.isoformat(),
            "market": "Dutch TTF natural gas, front-month future (€/MWh)",
            "trainingSamples": len(training_pool) - len(scored),
            "validationSamples": len(scored),
            "iterations": iterations,
            "lookbackDays": lookback_days,
            "historyStart": usable[0]["date"].isoformat(),
            "metrics": _metrics(actual_prices, model_prices),
            "baselineMetrics": _metrics(actual_prices, random_walk_prices),
            "directionHitRatePercent": direction_hits / len(scored) * 100,
            "directionPValue": _binomial_p_value(direction_hits, len(scored)),
            "returnCorrelation": return_correlation,
            "validation": validation,
            "history": history,
            "lossHistory": loss_history,
            "trading": trading_simulation,
            "policyComparison": comparison_policies,
            "signal": signal,
            "weatherForecast": _fetch_forecast_weather_daily(8),
            "validationMethod": "Daily expanding-window walk-forward",
            "weatherDataMethod": (
                f"Archived {FORECAST_HORIZON_DAYS}-day outlooks issued one day before "
                "each decision, plus their revision against the outlook issued a day earlier"
            ),
            "rollDaysExcluded": sum(point["rollDay"] for point in validation),
            "locations": list(GAS_LOCATIONS.keys()),
            "sources": [
                "Yahoo Finance (TTF=F, NG=F, BZ=F, EURUSD=X)",
                "Open-Meteo weather APIs",
            ],
        }
    finally:
        _training_lock.release()
