import React from "react";

export default function InfoPanel({ state }) {
  const batteryRatio = state ? Math.max(0, Math.min(1, state.battery / state.battery_capacity)) : 0;

  return (
    <section className="info-panel">
      <div>
        <span>Battery</span>
        <strong>
          {state?.battery ?? 0}/{state?.battery_capacity ?? 100}
        </strong>
        <div className="meter">
          <div style={{ width: `${batteryRatio * 100}%` }} />
        </div>
      </div>
      <div className="metric-grid">
        <Metric label="Step" value={state?.step_count ?? 0} />
        <Metric label="Last reward" value={(state?.reward ?? 0).toFixed(1)} />
        <Metric label="Episode return" value={(state?.episode_return ?? 0).toFixed(1)} />
        <Metric label="Last action" value={state?.action ?? "Reset"} />
        <Metric label="Dirt" value={state?.total_dirt ?? 0} />
        <Metric label="Robot" value={state?.robot_pos ? state.robot_pos.join(", ") : "-"} />
      </div>
      {state?.battery_reset ? (
        <div className="event-banner">
          Battery depleted. Robot relocated to dock.
        </div>
      ) : null}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
