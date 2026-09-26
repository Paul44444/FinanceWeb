import {
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";

export interface LiveLossPoint {
    step: number;
    loss: number;
}

interface LiveLossChartProps {
    data: LiveLossPoint[];
}

function LiveLossChart({ data }: LiveLossChartProps) {
    return (
        <section className="live-training">
            <h3>Live training loss</h3>

            <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />

                        <XAxis
                            dataKey="step"
                            label={{
                                value: "Data update",
                                position: "insideBottom",
                                offset: -5,
                            }}
                        />

                        <YAxis domain={["auto", "auto"]} />

                        <Tooltip />

                        <Line
                            type="monotone"
                            dataKey="loss"
                            name="Loss"
                            stroke="#dc2626"
                            isAnimationActive={true}
                            animationDuration={700}
                            animationEasing="ease-out"
                            dot={false}
                            activeDot={{ r: 5 }}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </section>
    );
}

export default LiveLossChart;
