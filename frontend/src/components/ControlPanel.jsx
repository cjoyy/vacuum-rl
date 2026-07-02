const ACTIONS = ["auto", "N", "S", "E", "W", "Stay", "Clean", "Charge"];

export default function ControlPanel({
  algorithms,
  selectedAlgorithm,
  selectedAction,
  speed,
  isPlaying,
  connectionStatus,
  onAlgorithmChange,
  onActionChange,
  onSpeedChange,
  onReset,
  onStep,
  onPlayPause,
}) {
  return (
    <section className="tool-panel">
      <div className="panel-row">
        <label>
          Algorithm
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
        Step action
        <select value={selectedAction} onChange={(event) => onActionChange(event.target.value)}>
          {ACTIONS.map((action) => (
            <option key={action} value={action}>
              {action === "auto" ? "Auto policy" : action}
            </option>
          ))}
        </select>
      </label>

      <div className="button-row">
        <button type="button" onClick={onReset}>
          Reset
        </button>
        <button type="button" onClick={onPlayPause}>
          {isPlaying ? "Pause" : "Play"}
        </button>
        <button type="button" onClick={onStep}>
          Step
        </button>
      </div>

      <label>
        Speed
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
