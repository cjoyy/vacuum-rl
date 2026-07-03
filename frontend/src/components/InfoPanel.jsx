import React from "react";
import { useTranslation } from "react-i18next";

const LOW_BATTERY_RATIO = 0.25;
const CRITICAL_BATTERY_RATIO = 0.1;

export default function InfoPanel({ state }) {
  const { t } = useTranslation();
  const batteryRatio = state ? Math.max(0, Math.min(1, state.battery / state.battery_capacity)) : 0;
  const isLow = batteryRatio > 0 && batteryRatio <= LOW_BATTERY_RATIO;
  const isCritical = batteryRatio > 0 && batteryRatio <= CRITICAL_BATTERY_RATIO;

  const meterClass =
    isCritical ? "meter meter-critical"
    : isLow ? "meter meter-low"
    : "meter";

  const fillClass =
    isCritical ? "fill fill-critical"
    : isLow ? "fill fill-low"
    : "fill";

  return (
    <section className="info-panel">
      <div>
        <span>{t("info.battery")}</span>
        <strong className={isLow ? "battery-warning" : ""}>
          {state?.battery ?? 0}/{state?.battery_capacity ?? 100}
        </strong>
        <div className={meterClass}>
          <div className={fillClass} style={{ width: `${batteryRatio * 100}%` }} />
        </div>
        {isLow && !state?.battery_reset ? (
          <div className="battery-low-warning">{t("info.battery_low")}</div>
        ) : null}
      </div>
      <div className="metric-grid">
        <Metric label={t("info.step")} value={state?.step_count ?? 0} />
        <Metric label={t("info.last_reward")} value={(state?.reward ?? 0).toFixed(1)} />
        <Metric label={t("info.episode_return")} value={(state?.episode_return ?? 0).toFixed(1)} />
        <Metric label={t("info.last_action")} value={state?.action ?? "Reset"} />
        <Metric label={t("info.dirt")} value={state?.total_dirt ?? 0} />
        <Metric label={t("info.robot")} value={state?.robot_pos ? state.robot_pos.join(", ") : "-"} />
      </div>
      {state?.battery_reset ? (
        <div className="event-banner event-banner-depletion">
          {t("info.battery_reset")}
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
