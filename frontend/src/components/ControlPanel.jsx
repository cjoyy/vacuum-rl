import React from "react";
import { useTranslation } from "react-i18next";

const ACTIONS = ["auto", "N", "S", "E", "W", "Stay", "Clean", "Charge"];

export default function ControlPanel({
  algorithms,
  selectedAlgorithm,
  selectedAction,
  speed,
  isPlaying,
  connectionStatus,
  canSend,
  onAlgorithmChange,
  onActionChange,
  onSpeedChange,
  onReset,
  onStep,
  onPlayPause,
}) {
  const { t } = useTranslation();

  return (
    <section className="tool-panel">
      <div className="panel-row">
        <label>
          {t("control.algorithm")}
          <select value={selectedAlgorithm} onChange={(event) => onAlgorithmChange(event.target.value)}>
            {algorithms.map((algorithm) => (
              <option key={algorithm.id} value={algorithm.id} disabled={!algorithm.available}>
                {algorithm.name}
              </option>
            ))}
          </select>
        </label>
        <span className={`status-dot ${connectionStatus === "connected" ? "online" : ""}`}>
          {connectionStatus}
        </span>
      </div>

      <label>
        {t("control.step_action")}
        <select value={selectedAction} onChange={(event) => onActionChange(event.target.value)}>
          {ACTIONS.map((action) => (
            <option key={action} value={action}>
              {action === "auto" ? t("control.auto_policy") : action}
            </option>
          ))}
        </select>
      </label>

      <div className="button-row">
        <button type="button" onClick={onReset}>
          {t("control.reset")}
        </button>
        <button type="button" onClick={onPlayPause} disabled={!canSend}>
          {isPlaying ? t("control.pause") : t("control.play")}
        </button>
        <button type="button" onClick={onStep} disabled={!canSend}>
          {t("control.step")}
        </button>
      </div>

      <label>
        {t("control.speed")}
        <input
          type="range"
          min="120"
          max="1200"
          step="40"
          value={speed}
          onChange={(event) => onSpeedChange(Number(event.target.value))}
        />
      </label>
    </section>
  );
}
