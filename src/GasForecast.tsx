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

type Instrument = "cfd" | "future";

interface ValidationPoint {
    date: string;
    nextDate: string;
    close: number;
    nextClose: number;
    tradable: boolean;
    rollDay: boolean;
    predicted: number | null;
    predictedChange: number | null;
}

interface GasPolicy {
    name: string;
    instrument: Instrument;
    startingBudget: number;
    finalCapital: number;
    profit: number;
    returnPercent: number;
    tradingProfitBeforeCosts: number;
    transactionCosts: number;
    financingCosts: number;
    leverage: number;
    spreadPerMWh: number;
    commissionPerMWh: number;
    financingRatePercent: number;
    marginPercent: number;
    contractSizeMWh: number;
    tradeCount: number;
    activeDays: number;
    winningDays: number;
    directionHitRatePercent: number;
    blockedSignals: number;
    meanDailyPnl: number;
    meanDailyPnlConfidence: number;
    sharpeRatio: number;
    maxDrawdownPercent: number;
    stoppedOut: boolean;
    trades: {
        time: string;
        action: "BUY" | "SELL";
        positionAfterMWh: number;
        quantityMWh: number;
        marketPrice: number;
        predictedChange: number | null;
        equityAfter: number;
    }[];
    equityCurve: { time: string; equity: number }[];
}

interface GasResult {
    trainedAt: string;
    market: string;
    trainingSamples: number;
    validationSamples: number;
    historyStart: string;
    metrics: { mae: number; rmse: number; correlation: number };
    baselineMetrics: { mae: number; rmse: number; correlation: number };
    directionHitRatePercent: number;
    directionPValue: number;
    returnCorrelation: number;
    validation: ValidationPoint[];
    lossHistory: { iteration: number; loss: number }[];
    trading: GasPolicy;
    policyComparison: GasPolicy[];
    signal: null | {
        decisionDate: string;
        lastClose: number;
        predictedReturnPercent: number;
        predictedClose: number;
        predictedChange: number;
        requiredEdge: number;
        action: "Long" | "Short" | "Stay flat";
        positionMWh: number;
        nextIsRollDay: boolean;
    };
    weatherForecast: { date: string; temperature: number; hdd: number; windSpeed: number }[];
    validationMethod: string;
    weatherDataMethod: string;
    rollDaysExcluded: number;
    locations: string[];
}

interface GasForecastProps {
    apiBaseUrl: string;
    backendReady: boolean;
}

const INSTRUMENT_DEFAULTS: Record<Instrument, {
    spread: number;
    commission: number;
    marginPercent: number;
    financingPercent: number;
}> = {
    cfd: { spread: 0.2, commission: 0, marginPercent: 10, financingPercent: 6 },
    future: { spread: 0.05, commission: 0.02, marginPercent: 20, financingPercent: 0 },
};

const formatDate = (value: string) =>
    new Intl.DateTimeFormat("de-DE", {
        day: "2-digit",
        month: "2-digit",
        year: "2-digit",
    }).format(new Date(value));

const signed = (value: number, digits = 2) =>
    `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;

function GasForecast({ apiBaseUrl, backendReady }: GasForecastProps) {
    const [lookbackDays, setLookbackDays] = useState(900);
    const [iterations, setIterations] = useState(300);
    const [testDays, setTestDays] = useState(60);
    const [startingBudget, setStartingBudget] = useState(1000);
    const [instrument, setInstrument] = useState<Instrument>("cfd");
    const [leverage, setLeverage] = useState(1);
    const [spread, setSpread] = useState(INSTRUMENT_DEFAULTS.cfd.spread);
    const [commission, setCommission] = useState(INSTRUMENT_DEFAULTS.cfd.commission);
    const [marginPercent, setMarginPercent] = useState(INSTRUMENT_DEFAULTS.cfd.marginPercent);
    const [financingPercent, setFinancingPercent] = useState(INSTRUMENT_DEFAULTS.cfd.financingPercent);
    const [contractSize, setContractSize] = useState(730);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState<{ stage: string; step: number; total: number } | null>(null);
    const [error, setError] = useState("");
    const [result, setResult] = useState<GasResult | null>(null);

    function selectInstrument(value: Instrument) {
        const defaults = INSTRUMENT_DEFAULTS[value];
        setInstrument(value);
        setSpread(defaults.spread);
        setCommission(defaults.commission);
        setMarginPercent(defaults.marginPercent);
        setFinancingPercent(defaults.financingPercent);
    }

    function trainModel() {
        setLoading(true);
        setError("");
        setProgress(null);
        const parameters = new URLSearchParams({
            lookback_days: String(lookbackDays),
            iterations: String(iterations),
            test_days: String(testDays),
            starting_budget: String(startingBudget),
            instrument,
            leverage: String(leverage),
            spread_per_mwh: String(spread),
            commission_per_mwh: String(commission),
            financing_rate: String(instrument === "cfd" ? financingPercent / 100 : 0),
            margin_rate: String(marginPercent / 100),
            contract_size_mwh: String(contractSize),
        });
        const eventSource = new EventSource(`${apiBaseUrl}/api/gas/train-stream?${parameters}`);

        eventSource.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === "progress") {
                setProgress(message);
            }
            if (message.type === "complete") {
                setResult(message.data);
                setLoading(false);
                eventSource.close();
            }
            if (message.type === "error") {
                setError(message.message);
                setLoading(false);
                eventSource.close();
            }
        };
        eventSource.onerror = () => {
            setError("The gas training connection was interrupted.");
            setLoading(false);
            eventSource.close();
        };
    }

    const modelBeatsRandomWalk = result && result.metrics.mae < result.baselineMetrics.mae;
    const pnlInterval = result && {
        low: result.trading.meanDailyPnl - result.trading.meanDailyPnlConfidence,
        high: result.trading.meanDailyPnl + result.trading.meanDailyPnlConfidence,
    };

    return (
        <section className="electricity-forecast">
            <div className="section-intro">
                <h2>Natural gas price forecast (TTF)</h2>
                <p>
                    Predicts the next-day move of the Dutch TTF front-month future, the benchmark
                    for German wholesale gas. Inputs: recent TTF returns and volatility, the previous
                    day's Henry Hub, Brent and EUR/USD moves, and north-west European heating-degree-day
                    and wind outlooks including how those outlooks were revised.
                </p>
            </div>

            <div className="assumption-summary">
                <strong>Market access for private traders.</strong> Physical spot gas at the German
                hub (THE) requires exchange membership and a balancing group. Private investors usually
                reach TTF through CFDs (ESMA retail limit 10:1 leverage, negative-balance protection)
                or through a futures broker, where one ICE Endex TTF contract covers 1 MW for every hour
                of the delivery month (about 720–744 MWh). Costs below are adjustable assumptions, not
                broker quotes.
            </div>

            <div className="forecast-controls">
                <label>
                    History (days)
                    <input type="number" min="200" max="1200" value={lookbackDays}
                        onChange={(event) => setLookbackDays(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Training iterations
                    <input type="number" min="50" max="2000" value={iterations}
                        onChange={(event) => setIterations(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Walk-forward test (trading days)
                    <input type="number" min="10" max="250" value={testDays}
                        onChange={(event) => setTestDays(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Simulated budget (€)
                    <input type="number" min="100" max="1000000" value={startingBudget}
                        onChange={(event) => setStartingBudget(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Instrument
                    <select value={instrument}
                        onChange={(event) => selectInstrument(event.target.value as Instrument)} disabled={loading}>
                        <option value="cfd">CFD (fractional size)</option>
                        <option value="future">Exchange future (whole contracts)</option>
                    </select>
                </label>
                <label>
                    Leverage (exposure / equity)
                    <input type="number" min="0.1" max="10" step="0.1" value={leverage}
                        onChange={(event) => setLeverage(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Bid-ask spread (€/MWh)
                    <input type="number" min="0" max="10" step="0.01" value={spread}
                        onChange={(event) => setSpread(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Commission (€/MWh per side)
                    <input type="number" min="0" max="10" step="0.005" value={commission}
                        onChange={(event) => setCommission(Number(event.target.value))} disabled={loading} />
                </label>
                <label>
                    Initial margin (%)
                    <input type="number" min="1" max="100" step="1" value={marginPercent}
                        onChange={(event) => setMarginPercent(Number(event.target.value))} disabled={loading} />
                </label>
                {instrument === "cfd" ? (
                    <label>
                        Overnight financing (%/year)
                        <input type="number" min="0" max="50" step="0.5" value={financingPercent}
                            onChange={(event) => setFinancingPercent(Number(event.target.value))} disabled={loading} />
                    </label>
                ) : (
                    <label>
                        Contract size (MWh)
                        <input type="number" min="0.01" max="10000" step="1" value={contractSize}
                            onChange={(event) => setContractSize(Number(event.target.value))} disabled={loading} />
                    </label>
                )}
                <button type="button" onClick={trainModel} disabled={loading || !backendReady}>
                    {!backendReady
                        ? "Connecting backend..."
                        : loading
                        ? "Training..."
                        : "Train and backtest"}
                </button>
            </div>

            {loading && progress && (
                <p className="chart-note">
                    {progress.stage}
                    {progress.total > 0 && ` · day ${progress.step} of ${progress.total}`}
                    {progress.total > 0 && (
                        <progress value={progress.step} max={progress.total} style={{ width: "100%" }} />
                    )}
                </p>
            )}

            {error && <div className="error">{error}</div>}

            {result && (
                <>
                    <div className="metric-grid">
                        <article>
                            <span>Model MAE (next close)</span>
                            <strong>{result.metrics.mae.toFixed(2)} €/MWh</strong>
                        </article>
                        <article>
                            <span>"Tomorrow = today" MAE</span>
                            <strong>{result.baselineMetrics.mae.toFixed(2)} €/MWh</strong>
                        </article>
                        <article>
                            <span>Direction hit rate</span>
                            <strong>{result.directionHitRatePercent.toFixed(1)}%</strong>
                        </article>
                        <article>
                            <span>Return correlation</span>
                            <strong>{result.returnCorrelation.toFixed(3)}</strong>
                        </article>
                        <article>
                            <span>Test days</span>
                            <strong>{result.validationSamples}</strong>
                        </article>
                    </div>

                    <p className="chart-note">
                        {modelBeatsRandomWalk
                            ? "The model was slightly more accurate than assuming no change. "
                            : "The model was not more accurate than assuming no change. "}
                        A hit rate of {result.directionHitRatePercent.toFixed(1)}% over {result.validationSamples} days
                        has a p-value of {result.directionPValue.toFixed(2)} against a coin flip
                        {result.directionPValue < 0.05 ? " (statistically significant)." : " (not statistically significant)."}
                        {" "}Training data starts {formatDate(result.historyStart)} ({result.trainingSamples} days before the test).
                    </p>

                    {result.signal && (
                        <div className="assumption-summary benchmark-summary">
                            <strong>Simulated signal after the {formatDate(result.signal.decisionDate)} close</strong>
                            <span>
                                Last settlement {result.signal.lastClose.toFixed(2)} €/MWh; model expects
                                {" "}{signed(result.signal.predictedReturnPercent)}%
                                ({result.signal.predictedClose.toFixed(2)} €/MWh). Costs require a move of at least
                                {" "}{result.signal.requiredEdge.toFixed(2)} €/MWh.
                                {" "}Simulated action: <strong>{result.signal.action}</strong>
                                {result.signal.positionMWh > 0 && ` (${result.signal.positionMWh.toFixed(2)} MWh)`}
                                {result.signal.action !== "Stay flat" && result.signal.positionMWh === 0 &&
                                    ", but the budget does not cover the margin for one contract"}.
                                {result.signal.nextIsRollDay && " The next session rolls to a new front month, so the policy stays flat."}
                                {" "}No real order is placed.
                            </span>
                        </div>
                    )}

                    <h3>Out-of-sample validation</h3>
                    <p className="chart-note">
                        {result.validationMethod}: each test day is predicted by a newly trained model that only
                        sees outcomes known at that day's close. Weather inputs are {result.weatherDataMethod.toLowerCase()}.
                        {" "}{result.rollDaysExcluded} contract-roll days were excluded because the continuous
                        front-month series jumps between two contracts there.
                    </p>
                    <div className="chart-container electricity-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.validation}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="nextDate" tickFormatter={formatDate} minTickGap={45} />
                                <YAxis unit=" €" domain={["auto", "auto"]} />
                                <Tooltip labelFormatter={(value) => formatDate(String(value))}
                                    formatter={(value) => (typeof value === "number" ? value.toFixed(2) : value)} />
                                <Legend />
                                <Line dataKey="nextClose" name="Actual settlement €/MWh" stroke="#111827" dot={false} strokeWidth={2} />
                                <Line dataKey="predicted" name="Model" stroke="#7c3aed" dot={false} strokeWidth={2} />
                                <Line dataKey="close" name="Tomorrow = today" stroke="#9ca3af" dot={false} strokeDasharray="5 5" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <h3>Weather outlook (north-west Europe)</h3>
                    <div className="chart-container electricity-chart compact-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.weatherForecast}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="date" tickFormatter={formatDate} />
                                <YAxis yAxisId="heat" />
                                <YAxis yAxisId="wind" orientation="right" />
                                <Tooltip labelFormatter={(value) => formatDate(String(value))}
                                    formatter={(value) => (typeof value === "number" ? value.toFixed(1) : value)} />
                                <Legend />
                                <Line yAxisId="heat" dataKey="hdd" name="Heating degree days" stroke="#ef4444" dot={false} strokeWidth={2} />
                                <Line yAxisId="heat" dataKey="temperature" name="Temperature °C" stroke="#f59e0b" dot={false} />
                                <Line yAxisId="wind" dataKey="windSpeed" name="Wind 100 m km/h" stroke="#2563eb" dot={false} />
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

                    <h3>Simulated trading</h3>
                    <p className="chart-note">
                        Simulation only. After each settlement the policy goes long, short or flat for the next
                        session when the predicted move exceeds spread, commission and financing. Positions are
                        closed before every contract roll.
                    </p>
                    <div className="metric-grid trading-metrics">
                        <article>
                            <span>Capital after simulation</span>
                            <strong>{result.trading.finalCapital.toFixed(2)} €</strong>
                        </article>
                        <article>
                            <span>Net profit</span>
                            <strong className={result.trading.profit >= 0 ? "positive" : "negative"}>
                                {signed(result.trading.profit)} €
                            </strong>
                        </article>
                        <article>
                            <span>Return</span>
                            <strong>{result.trading.returnPercent.toFixed(2)}%</strong>
                        </article>
                        <article>
                            <span>Winning days</span>
                            <strong>{result.trading.winningDays} / {result.trading.activeDays}</strong>
                        </article>
                        <article>
                            <span>Maximum drawdown</span>
                            <strong>{result.trading.maxDrawdownPercent.toFixed(2)}%</strong>
                        </article>
                    </div>

                    <div className="assumption-summary">
                        {result.trading.instrument === "cfd" ? "CFD" : "Exchange future"}: leverage
                        {" "}{result.trading.leverage.toFixed(1)}×, margin {result.trading.marginPercent.toFixed(0)}%,
                        {" "}spread {result.trading.spreadPerMWh.toFixed(3)} €/MWh, commission
                        {" "}{result.trading.commissionPerMWh.toFixed(3)} €/MWh per side
                        {result.trading.instrument === "cfd"
                            ? `, financing ${result.trading.financingRatePercent.toFixed(1)}%/year`
                            : `, contract size ${result.trading.contractSizeMWh.toFixed(0)} MWh`}.
                        {" "}Gross trading result {signed(result.trading.tradingProfitBeforeCosts)} €, minus
                        {" "}{result.trading.transactionCosts.toFixed(2)} € transaction costs and
                        {" "}{result.trading.financingCosts.toFixed(2)} € financing.
                        {pnlInterval && result.trading.activeDays > 1 && (
                            <>
                                {" "}Mean P&L per active day {signed(result.trading.meanDailyPnl)} € (95% interval
                                {" "}{signed(pnlInterval.low)} to {signed(pnlInterval.high)} €
                                {pnlInterval.low <= 0 && pnlInterval.high >= 0 ? ", includes zero" : ""}).
                            </>
                        )}
                        {result.trading.blockedSignals > 0 && (
                            <> {result.trading.blockedSignals} signals could not be traded because the budget does not cover the margin for one contract.</>
                        )}
                        {result.trading.stoppedOut && " The account was wiped out and trading stopped."}
                    </div>

                    <div className="trade-table-wrap">
                        <table className="trade-table comparison-table">
                            <thead>
                                <tr><th>Policy</th><th>Profit</th><th>Return</th><th>Sharpe</th><th>Max drawdown</th><th>Hit rate</th><th>Trades</th></tr>
                            </thead>
                            <tbody>
                                {result.policyComparison.map((policy) => (
                                    <tr key={policy.name}>
                                        <td>{policy.name}</td>
                                        <td>{signed(policy.profit)} €</td>
                                        <td>{policy.returnPercent.toFixed(2)}%</td>
                                        <td>{policy.sharpeRatio.toFixed(2)}</td>
                                        <td>{policy.maxDrawdownPercent.toFixed(1)}%</td>
                                        <td>{policy.directionHitRatePercent.toFixed(1)}%</td>
                                        <td>{policy.tradeCount}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <div className="chart-container electricity-chart compact-chart">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={result.trading.equityCurve}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="time" tickFormatter={formatDate} minTickGap={45} />
                                <YAxis unit=" €" domain={["auto", "auto"]} />
                                <Tooltip labelFormatter={(value) => formatDate(String(value))}
                                    formatter={(value) => (typeof value === "number" ? value.toFixed(2) : value)} />
                                <Line dataKey="equity" name="Simulated equity" stroke="#0891b2" dot={false} strokeWidth={2} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    {result.trading.trades.length > 0 && (
                        <div className="trade-table-wrap">
                            <table className="trade-table">
                                <thead>
                                    <tr><th>Date</th><th>Action</th><th>Settlement</th><th>Predicted move</th><th>Quantity</th><th>Position after</th><th>Equity after</th></tr>
                                </thead>
                                <tbody>
                                    {result.trading.trades.map((trade, index) => (
                                        <tr key={`${trade.time}-${index}`}>
                                            <td>{formatDate(trade.time)}</td>
                                            <td className={trade.action === "BUY" ? "buy-action" : "sell-action"}>{trade.action}</td>
                                            <td>{trade.marketPrice.toFixed(2)} €/MWh</td>
                                            <td>{trade.predictedChange === null ? "close" : `${signed(trade.predictedChange)} €/MWh`}</td>
                                            <td>{trade.quantityMWh.toFixed(2)} MWh</td>
                                            <td>{trade.positionAfterMWh.toFixed(2)} MWh</td>
                                            <td>{trade.equityAfter.toFixed(2)} €</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}

                    <p className="data-attribution">
                        Prices: Yahoo Finance (TTF=F continuous front month, NG=F, BZ=F, EURUSD=X).
                        Weather: Open-Meteo, averaged over {result.locations.join(", ")}.
                    </p>
                </>
            )}
        </section>
    );
}

export default GasForecast;
