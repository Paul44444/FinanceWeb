import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import NetworkCharts from "./NetworkCharts";
import ElectricityForecast from "./ElectricityForecast";

import LiveLossChart from "./LiveLossChart";
import type { LiveLossPoint } from "./LiveLossChart";
//import { API_BASE_URL } from "./config";

interface CalculationResponse {
    result: number;
}

interface NetworkResult {
    stock: string;
    iterations: number;
    losses: number[];
    losses_simple: number[];
    cash_history: number[];
    cash_history_linear: number[];
    overperform_simple: number[];
    overperform_linear: number[];
}

function App() {
    const [activeView, setActiveView] = useState<"stocks" | "electricity">("electricity");
    const [x, setX] = useState<number>(5);
    const [y, setY] = useState<number>(7);
    const [result, setResult] = useState<number | null>(null);
    const [error, setError] = useState<string>("");
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const [liveLossData, setLiveLossData] =
        useState<LiveLossPoint[]>([]);

    const [trainingProgress, setTrainingProgress] =
        useState<number>(0);

    const [networkResult, setNetworkResult] =
        useState<NetworkResult | null>(null);

    const [networkLoading, setNetworkLoading] =
        useState<boolean>(false);
    const [trainingIterations, setTrainingIterations] =
        useState<number>(200);
    const [apiBaseUrl, setApiBaseUrl] = useState<string>("");
    const [backendReady, setBackendReady] = useState<boolean>(import.meta.env.DEV);

    useEffect(() => {
        if (import.meta.env.DEV) {
            return;
        }

        fetch("/backend.json", { cache: "no-store" })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`Backend configuration returned ${response.status}`);
                }
                return response.json();
            })
            .then((configuration: { url?: string }) => {
                const url = configuration.url?.replace(/\/$/, "");
                if (!url?.startsWith("https://")) {
                    throw new Error("No public backend URL is configured.");
                }
                setApiBaseUrl(url);
                setBackendReady(true);
            })
            .catch((configurationError) => {
                console.error(configurationError);
                setError("The Python backend is currently unavailable.");
            });
    }, []);

    async function handleCalculate(event: FormEvent<HTMLFormElement>) {
        event.preventDefault();

        setIsLoading(true);
        setError("");

        try {
            const response = await fetch(`${apiBaseUrl}/api/calculate`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    x,
                    y,
                }),
            });

            if (!response.ok) {
                throw new Error(`Backend returned error ${response.status}`);
            }

            const data: CalculationResponse = await response.json();
            setResult(data.result);
        } catch (requestError) {
            console.error(requestError);
            setError(
                "Could not reach the Python backend. Make sure it is running.",
            );
        } finally {
            setIsLoading(false);
        }
    }

    function runNetwork() {
        setNetworkLoading(true);
        setNetworkResult(null);
        setLiveLossData([]);
        setTrainingProgress(0);
        setError("");

        const eventSource = new EventSource(
            `${apiBaseUrl}/api/run-network-stream?iterations=${trainingIterations}`,
        );

        console.log("Training stream URL:", eventSource.url);
        eventSource.onmessage = (event) => {
            const message = JSON.parse(event.data);

            if (message.type === "progress") {
                setLiveLossData((currentData) => [
                    ...currentData,
                    {
                        step: message.step,
                        loss: message.loss,
                    },
                ]);

                setTrainingProgress(
                    Math.round((message.step / message.total) * 100),
                );
            }

            if (message.type === "complete") {
                setNetworkResult(message.data);
                setTrainingProgress(100);
                setNetworkLoading(false);
                eventSource.close();
            }

            if (message.type === "error") {
                setError(message.message);
                setNetworkLoading(false);
                eventSource.close();
            }
        };

        eventSource.onerror = () => {
            setError("The live training connection was interrupted.");
            setNetworkLoading(false);
            eventSource.close();
        };
    }
    const SHOW_CALCULATOR = false;

    return (
        <main className="app">
            <section className="calculator">
                <h1>Python Finance Application</h1>
                <nav className="experiment-tabs" aria-label="Finance experiments">
                    <button
                        type="button"
                        className={activeView === "electricity" ? "active" : ""}
                        onClick={() => setActiveView("electricity")}
                    >
                        Electricity forecast
                    </button>
                    <button
                        type="button"
                        className={activeView === "stocks" ? "active" : ""}
                        onClick={() => setActiveView("stocks")}
                    >
                        Stock experiment
                    </button>
                </nav>

                {activeView === "electricity" ? (
                    <ElectricityForecast
                        apiBaseUrl={apiBaseUrl}
                        backendReady={backendReady}
                    />
                ) : (
                <>
                {SHOW_CALCULATOR && (<>
                <p>
                    Enter two values. The calculation will be performed by the
                    Python backend.
                </p>

                <form onSubmit={handleCalculate}>
                    <label>
                        Value x
                        <input
                            type="number"
                            value={x}
                            step="any"
                            onChange={(event) =>
                                setX(Number(event.target.value))
                            }
                        />
                    </label>

                    <label>
                        Value y
                        <input
                            type="number"
                            value={y}
                            step="any"
                            onChange={(event) =>
                                setY(Number(event.target.value))
                            }
                        />
                    </label>

                    <button type="submit" disabled={isLoading}>
                        {isLoading ? "Calculating..." : "Calculate x² + y²"}
                    </button>
                </form>

                {result !== null && (
                    <div className="result">
                        Result: <strong>{result}</strong>
                    </div>
                )}

                <hr />
                    </>
                )}

                <h2>Stock neural network</h2>

                <label className="training-iterations">
                    Training iterations
                    <input
                        type="number"
                        min="1"
                        max="10000"
                        value={trainingIterations}
                        onChange={(event) =>
                            setTrainingIterations(
                                Math.min(10000, Math.max(1, Number(event.target.value))),
                            )
                        }
                        disabled={networkLoading}
                    />
                </label>

                <button
                    type="button"
                    onClick={runNetwork}
                    disabled={networkLoading || !backendReady}
                >
                    {!backendReady
                        ? "Connecting backend..."
                        : networkLoading
                        ? "Training network..."
                        : "Run neural network"}
                </button>

                {(networkLoading || liveLossData.length > 0) && (
                    <section>
                        <p>
                            Training progress: {trainingProgress}% · iteration: {liveLossData.at(-1)?.step ?? 0} of {trainingIterations}
                            <span className="training-hint">
                                (the chart stops after the selected number of iterations)
                            </span>
                        </p>

                        <progress
                            value={trainingProgress}
                            max={100}
                            style={{ width: "100%" }}
                        />

                        <LiveLossChart data={liveLossData} />
                    </section>
                )}

                {networkResult && (
                    <section className="network-result">
                        <h2>Result for {networkResult.stock}</h2>

                        <p>
                            Final loss:{" "}
                            <strong>
                                {networkResult.losses.at(-1)?.toFixed(6) ?? "No data"}
                            </strong>
                        </p>

                        <p>
                            Final net worth:{" "}
                            <strong>
                                {networkResult.cash_history.at(-1)?.toFixed(6) ??
                                    "No data"}
                            </strong>
                        </p>

                        <NetworkCharts result={networkResult} />
                    </section>
                )}

                {error && <div className="error">{error}</div>}
                </>
                )}
            </section>
        </main>
    );
}

export default App;
