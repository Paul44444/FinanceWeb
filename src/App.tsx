import { useState } from "react";
import type { FormEvent } from "react";
import NetworkCharts from "./NetworkCharts";

import LiveLossChart from "./LiveLossChart";
import type { LiveLossPoint } from "./LiveLossChart";

interface CalculationResponse {
    result: number;
}

interface NetworkResult {
    stock: string;
    epochs: number;
    losses: number[];
    losses_simple: number[];
    cash_history: number[];
    cash_history_linear: number[];
    overperform_simple: number[];
    overperform_linear: number[];
}

function App() {
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
    const API_BASE_URL =
        import.meta.env.VITE_API_BASE_URL ?? "";

    async function handleCalculate(event: FormEvent<HTMLFormElement>) {
        event.preventDefault();

        setIsLoading(true);
        setError("");

        try {
            const response = await fetch("/api/calculate", {
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
            `${API_BASE_URL}/api/run-network-stream`,
        );

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

                <button
                    type="button"
                    onClick={runNetwork}
                    disabled={networkLoading}
                >
                    {networkLoading
                        ? "Training network..."
                        : "Run neural network"}
                </button>

                {(networkLoading || liveLossData.length > 0) && (
                    <section>
                        <p>Training progress: {trainingProgress}%</p>

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
            </section>
        </main>
    );
}

export default App;