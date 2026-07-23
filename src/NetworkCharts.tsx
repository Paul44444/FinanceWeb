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

interface NetworkChartsProps {
    result: NetworkResult;
}

function NetworkCharts({ result }: NetworkChartsProps) {
    const lossData = result.losses.map((loss, index) => ({
        step: index + 1,
        loss,
        absoluteLoss: result.losses_simple[index],
    }));

    const cashData = result.cash_history.map((cash, index) => ({
        step: index + 1,
        simpleStrategy: cash,
        linearStrategy: result.cash_history_linear[index],
    }));

    const overperformanceData = result.overperform_simple.map(
        (simple, index) => ({
            step: index + 1,
            simple,
            linear: result.overperform_linear[index],
        }),
    );

    return (
        <section className="charts">
            <h3>Training loss</h3>

            <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={lossData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                            dataKey="step"
                            label={{
                                value: "Training step",
                                position: "insideBottom",
                                offset: -5,
                            }}
                        />
                        <YAxis />
                        <Tooltip />
                        <Legend />

                        <Line
                            type="monotone"
                            dataKey="loss"
                            name="MSE loss"
                            stroke="#dc2626"
                            dot={false}
                        />

                        <Line
                            type="monotone"
                            dataKey="absoluteLoss"
                            name="Absolute loss"
                            stroke="#f59e0b"
                            dot={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <h3>Portfolio value</h3>

            <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={cashData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                            dataKey="step"
                            label={{
                                value: "Training step",
                                position: "insideBottom",
                                offset: -5,
                            }}
                        />
                        <YAxis />
                        <Tooltip />
                        <Legend />

                        <Line
                            type="monotone"
                            dataKey="simpleStrategy"
                            name="Simple strategy"
                            stroke="#2563eb"
                            dot={false}
                        />

                        <Line
                            type="monotone"
                            dataKey="linearStrategy"
                            name="Linear strategy"
                            stroke="#16a34a"
                            dot={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <h3>Overperformance</h3>

            <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={overperformanceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                            dataKey="step"
                            label={{
                                value: "Training step",
                                position: "insideBottom",
                                offset: -5,
                            }}
                        />
                        <YAxis />
                        <Tooltip />
                        <Legend />

                        <Line
                            type="monotone"
                            dataKey="simple"
                            name="Simple strategy"
                            stroke="#7c3aed"
                            dot={false}
                        />

                        <Line
                            type="monotone"
                            dataKey="linear"
                            name="Linear strategy"
                            stroke="#0891b2"
                            dot={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </section>
    );
}

export default NetworkCharts;