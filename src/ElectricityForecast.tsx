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
                        The last {result.validationSamples} known hours were held out
                        from training. The baseline repeats the price from 24 hours earlier.
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
