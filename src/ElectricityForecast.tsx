import { useState } from "react";
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

interface PricePoint {
    time: string;
    actual?: number | null;
    predicted: number;
    baseline?: number;
}

interface ForecastPoint extends PricePoint {
    temperature: number;
    windSpeed: number;
    solarRadiation: number;
    cloudCover: number;
}

interface ElectricityResult {
    trainedAt: string;
    trainingSamples: number;
    validationSamples: number;
    iterations: number;
    forecastHours: number;
    metrics: { mae: number; rmse: number; correlation: number };
    baselineMetrics: { mae: number; rmse: number; correlation: number };
    validation: PricePoint[];
    forecast: ForecastPoint[];
    lossHistory: { iteration: number; loss: number }[];
    locations: string[];
    validationMethod: string;
    weatherDataMethod: string;
    trading: {
        name: string;
        startingBudget: number;
        batteryInvestment: number;
        investedCapital: number;
        finalCapital: number;
        finalCash: number;
        profit: number;
        returnPercent: number;
        capacityMWh: number;
        roundTripEfficiency: number;
        powerMW: number;
        marketFeePerMWh: number;
        degradationCostPerMWh: number;
        selfDischargePercentPerDay: number;
        completedCycles: number;
        winningCycles: number;
        maxDrawdownPercent: number;
        trades: {
            time: string;
            action: "BUY" | "SELL";
            marketPrice: number;
            predictedPrice: number;
            energyMWh: number;
            cashAfter: number;
            cycleProfit?: number;
        }[];
        equityCurve: { time: string; equity: number }[];
    };
    policyComparison: {
        name: string;
        profit: number;
        returnPercent: number;
        completedCycles: number;
        winningCycles: number;
    }[];
    plannedTrades: {
        buyTime: string;
        sellTime: string;
        predictedBuyPrice: number;
        predictedSellPrice: number;
        expectedProfit: number;
    }[];
}

interface ElectricityForecastProps {
    apiBaseUrl: string;
    backendReady: boolean;
}

const formatTime = (value: string) =>
    new Intl.DateTimeFormat("de-DE", {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
    }).format(new Date(value));

function ElectricityForecast({ apiBaseUrl, backendReady }: ElectricityForecastProps) {
    const [lookbackDays, setLookbackDays] = useState(60);
    const [iterations, setIterations] = useState(800);
    const [forecastHours, setForecastHours] = useState(48);
    const [startingBudget, setStartingBudget] = useState(1000);
    const [storageCapacity, setStorageCapacity] = useState(1);
    const [efficiencyPercent, setEfficiencyPercent] = useState(90);
    const [chargePower, setChargePower] = useState(0.5);
    const [batteryCost, setBatteryCost] = useState(400);
    const [marketFee, setMarketFee] = useState(3);
    const [degradationCost, setDegradationCost] = useState(20);
    const [selfDischarge, setSelfDischarge] = useState(0.2);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [result, setResult] = useState<ElectricityResult | null>(null);

    async function trainModel() {
        setLoading(true);
        setError("");
        try {
            const response = await fetch(`${apiBaseUrl}/api/electricity/train`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    lookback_days: lookbackDays,
                    iterations,
                    forecast_hours: forecastHours,
                    starting_budget: startingBudget,
                    storage_capacity_mwh: storageCapacity,
                    round_trip_efficiency: efficiencyPercent / 100,
                    charge_power_mw: chargePower,
                    battery_cost_per_kwh: batteryCost,
                    market_fee_per_mwh: marketFee,
                    degradation_cost_per_mwh: degradationCost,
                    self_discharge_percent_per_day: selfDischarge,
                }),
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail ?? `Backend returned ${response.status}`);
            }
            setResult(payload);
        } catch (requestError) {
            setError(
                requestError instanceof Error
                    ? requestError.message
                    : "Electricity-price training failed.",
            );
        } finally {
            setLoading(false);
        }
    }

    return (
        <section className="electricity-forecast">
            <div className="section-intro">
                <h2>German electricity price forecast</h2>
                <p>
                    A first weather-aware model for the DE-LU day-ahead market.
                    It combines calendar cycles, temperature, wind at 100 m,
                    solar radiation, cloud cover and price lags.
                </p>
            </div>

            <div className="forecast-controls">
                <label>
                    History (days)
                    <input
                        type="number"
                        min="21"
                        max="365"
                        value={lookbackDays}
                        onChange={(event) => setLookbackDays(Number(event.target.value))}
                        disabled={loading}
                    />
                </label>
                <label>
                    Training iterations
                    <input
                        type="number"
                        min="50"
                        max="5000"
                        value={iterations}
                        onChange={(event) => setIterations(Number(event.target.value))}
                        disabled={loading}
                    />
                </label>
                <label>
                    Forecast (hours)
                    <input
                        type="number"
                        min="12"
                        max="72"
                        value={forecastHours}
                        onChange={(event) => setForecastHours(Number(event.target.value))}
                        disabled={loading}
                    />
                </label>
                <label>
                    Simulated budget (€)
                    <input
                        type="number"
                        min="100"
                        max="1000000"
                        value={startingBudget}
                        onChange={(event) => setStartingBudget(Number(event.target.value))}
                        disabled={loading}
                    />
                </label>
                <label>
                    Storage capacity (MWh)
                    <input
                        type="number"
                        min="0.01"
                        max="100"
                        step="0.01"
                        value={storageCapacity}
                        onChange={(event) => setStorageCapacity(Number(event.target.value))}
                        disabled={loading}
                    />
                </label>
                <label>
                    Round-trip efficiency (%)
                    <input
                        type="number"
                        min="50"
                        max="100"
                        value={efficiencyPercent}
                        onChange={(event) => setEfficiencyPercent(Number(event.target.value))}
                        disabled={loading}
                    />
                </label>
                <label>
                    Charge/discharge power (MW)
                    <input type="number" min="0.01" max="100" step="0.01" value={chargePower}
                        onChange={(event) => setChargePower(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Battery investment (€/kWh)
                    <input type="number" min="0" max="5000" step="10" value={batteryCost}
                        onChange={(event) => setBatteryCost(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Market/grid fee (€/MWh/leg)
                    <input type="number" min="0" max="500" step="0.5" value={marketFee}
                        onChange={(event) => setMarketFee(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Battery wear (€/MWh cycle)
                    <input type="number" min="0" max="1000" step="1" value={degradationCost}
                        onChange={(event) => setDegradationCost(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Self-discharge (%/day)
                    <input type="number" min="0" max="20" step="0.1" value={selfDischarge}
                        onChange={(event) => setSelfDischarge(Number(event.target.value))} disabled={loading} />
                </label>
                <button
                    type="button"
                    onClick={trainModel}
                    disabled={loading || !backendReady}
                >
                    {!backendReady
                        ? "Connecting backend..."
                        : loading
                        ? "Downloading data and training..."
                        : "Train and forecast"}
                </button>
            </div>

            {error && <div className="error">{error}</div>}

            {result && (
                <>
                    <div className="metric-grid">
                        <article>
                            <span>Validation MAE</span>
                            <strong>{result.metrics.mae.toFixed(2)} €/MWh</strong>
                        </article>
                        <article>
                            <span>24 h baseline MAE</span>
                            <strong>{result.baselineMetrics.mae.toFixed(2)} €/MWh</strong>
                        </article>
                        <article>
                            <span>Correlation</span>
                            <strong>{result.metrics.correlation.toFixed(3)}</strong>
                        </article>
                        <article>
                            <span>Training observations</span>
                            <strong>{result.trainingSamples}</strong>
                        </article>
                    </div>

                    <h3>Out-of-sample validation</h3>
                    <p className="chart-note">
                        {result.validationMethod}: every test day is predicted by a newly
                        trained model that only sees earlier prices. Weather inputs are the
                        {" "}{result.weatherDataMethod.toLowerCase()}. The baseline repeats
                        the price from 24 hours earlier.
                    </p>
                    <div className="chart-container electricity-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.validation}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" tickFormatter={formatTime} minTickGap={45} />
                                <YAxis unit=" €" />
                                <Tooltip labelFormatter={(value) => formatTime(String(value))} />
                                <Legend />
                                <Line dataKey="actual" name="Actual €/MWh" stroke="#111827" dot={false} strokeWidth={2} />
                                <Line dataKey="predicted" name="Model" stroke="#7c3aed" dot={false} strokeWidth={2} />
                                <Line dataKey="baseline" name="24 h baseline" stroke="#9ca3af" dot={false} strokeDasharray="5 5" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <h3>Electricity price forecast</h3>
                    <div className="chart-container electricity-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.forecast}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" tickFormatter={formatTime} minTickGap={45} />
                                <YAxis unit=" €" />
                                <Tooltip labelFormatter={(value) => formatTime(String(value))} />
                                <Legend />
                                <Line dataKey="predicted" name="Forecast €/MWh" stroke="#dc2626" dot={false} strokeWidth={2} />
                                <Line dataKey="actual" name="Published market price" stroke="#111827" dot={false} connectNulls />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <h3>Weather drivers for the forecast</h3>
                    <div className="chart-container electricity-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.forecast}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" tickFormatter={formatTime} minTickGap={45} />
                                <YAxis yAxisId="weather" />
                                <YAxis yAxisId="radiation" orientation="right" />
                                <Tooltip labelFormatter={(value) => formatTime(String(value))} />
                                <Legend />
                                <Line yAxisId="weather" dataKey="temperature" name="Temperature °C" stroke="#ef4444" dot={false} />
                                <Line yAxisId="weather" dataKey="windSpeed" name="Wind 100 m km/h" stroke="#2563eb" dot={false} />
                                <Line yAxisId="radiation" dataKey="solarRadiation" name="Solar W/m²" stroke="#f59e0b" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <h3>Training loss</h3>
                    <div className="chart-container electricity-chart compact-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.lossHistory}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="iteration" />
                                <YAxis />
                                <Tooltip />
                                <Line dataKey="loss" name="Smooth L1 loss" stroke="#16a34a" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <h3>Simulated battery trading policy</h3>
                    <p className="chart-note">
                        Simulation only. Each day the policy selects one forecast-based
                        charge/discharge cycle and settles it against held-out market prices.
                        Return includes the battery investment and operating costs.
                    </p>
                    <div className="metric-grid trading-metrics">
                        <article>
                            <span>Capital after simulation</span>
                            <strong>{result.trading.finalCapital.toFixed(2)} €</strong>
                        </article>
                        <article>
                            <span>Operating profit</span>
                            <strong className={result.trading.profit >= 0 ? "positive" : "negative"}>
                                {result.trading.profit >= 0 ? "+" : ""}{result.trading.profit.toFixed(2)} €
                            </strong>
                        </article>
                        <article>
                            <span>Return on total capital</span>
                            <strong>{result.trading.returnPercent.toFixed(2)}%</strong>
                        </article>
                        <article>
                            <span>Winning cycles</span>
                            <strong>{result.trading.winningCycles} / {result.trading.completedCycles}</strong>
                        </article>
                        <article>
                            <span>Maximum drawdown</span>
                            <strong>{result.trading.maxDrawdownPercent.toFixed(2)}%</strong>
                        </article>
                    </div>

                    <div className="assumption-summary">
                        Total invested: {result.trading.investedCapital.toFixed(2)} €
                        {" "}(cash {result.trading.startingBudget.toFixed(2)} € + battery {result.trading.batteryInvestment.toFixed(2)} €).
                        Power limit {result.trading.powerMW.toFixed(2)} MW; market/grid fee {result.trading.marketFeePerMWh.toFixed(2)} €/MWh per leg;
                        {" "}wear {result.trading.degradationCostPerMWh.toFixed(2)} €/MWh per cycle;
                        {" "}self-discharge {result.trading.selfDischargePercentPerDay.toFixed(2)}%/day.
                    </div>

                    <div className="trade-table-wrap">
                        <table className="trade-table comparison-table">
                            <thead><tr><th>Policy</th><th>Profit</th><th>Return on capital</th><th>Winning cycles</th></tr></thead>
                            <tbody>
                                {result.policyComparison.map((policy) => (
                                    <tr key={policy.name}>
                                        <td>{policy.name}</td>
                                        <td>{policy.profit >= 0 ? "+" : ""}{policy.profit.toFixed(2)} €</td>
                                        <td>{policy.returnPercent.toFixed(3)}%</td>
                                        <td>{policy.winningCycles} / {policy.completedCycles}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <div className="chart-container electricity-chart compact-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.trading.equityCurve}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" tickFormatter={formatTime} minTickGap={45} />
                                <YAxis unit=" €" domain={["auto", "auto"]} />
                                <Tooltip labelFormatter={(value) => formatTime(String(value))} />
                                <Line dataKey="equity" name="Simulated equity" stroke="#0891b2" dot={false} strokeWidth={2} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="trade-table-wrap">
                        <table className="trade-table">
                            <thead>
                                <tr><th>Time</th><th>Action</th><th>Market</th><th>Forecast</th><th>Energy</th><th>Cash after</th></tr>
                            </thead>
                            <tbody>
                                {result.trading.trades.map((trade, index) => (
                                    <tr key={`${trade.time}-${trade.action}-${index}`}>
                                        <td>{formatTime(trade.time)}</td>
                                        <td className={trade.action === "BUY" ? "buy-action" : "sell-action"}>{trade.action}</td>
                                        <td>{trade.marketPrice.toFixed(2)} €/MWh</td>
                                        <td>{trade.predictedPrice.toFixed(2)} €/MWh</td>
                                        <td>{trade.energyMWh.toFixed(3)} MWh</td>
                                        <td>{trade.cashAfter.toFixed(2)} €</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {result.plannedTrades.length > 0 && (
                        <>
                            <h3>Next simulated forecast plan</h3>
                            <div className="planned-trades">
                                {result.plannedTrades.map((trade) => (
                                    <article key={`${trade.buyTime}-${trade.sellTime}`}>
                                        Buy {formatTime(trade.buyTime)} at predicted {trade.predictedBuyPrice.toFixed(2)} €/MWh,
                                        sell {formatTime(trade.sellTime)} at predicted {trade.predictedSellPrice.toFixed(2)} €/MWh.
                                        Expected simulated margin: {trade.expectedProfit.toFixed(2)} €.
                                    </article>
                                ))}
                            </div>
                        </>
                    )}

                    <p className="data-attribution">
                        Price data: Bundesnetzagentur | SMARD.de (CC BY 4.0).
                        Weather: Open-Meteo. Weather values are spatial averages for {result.locations.join(", ")}.
                    </p>
                </>
            )}
        </section>
    );
}

export default ElectricityForecast;
