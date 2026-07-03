import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { fetchAlgorithms, arenaWebsocketUrl } from "../api.js";
import MiniGridCanvas from "./MiniGridCanvas.jsx";

const MAX_INSTANCES = 5;
const MODEL_SEEDS = [12345, 12346, 12347, 12348, 12349];

export default function ArenaSection() {
  const { t } = useTranslation();
  const [algorithms, setAlgorithms] = useState([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState([]);
  const [arenaStates, setArenaStates] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [seed, setSeed] = useState(12345);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const socketRef = useRef(null);
  const reconnectDelayRef = useRef(1000);
  const maxReconnectDelay = 30000;
  const shouldReconnectRef = useRef(false);
  const reconnectTimerRef = useRef(null);

  useEffect(() => {
    fetchAlgorithms()
      .then((payload) => {
        setAlgorithms(payload.algorithms);

        const availableAlgorithms = payload.algorithms.filter((algorithm) => algorithm.available);
        const preferred = availableAlgorithms.find((algorithm) => algorithm.id === "ppo") || availableAlgorithms[0];
        if (preferred) {
          setSelectedAlgorithms((current) => (current.length > 0 ? current : [preferred.id]));
        }
      })
      .catch((err) => setError(err.message));
    return () => {
      shouldReconnectRef.current = false;
      window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (selectedAlgorithms.length === 0) {
      shouldReconnectRef.current = false;
      window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
      setArenaStates([]);
      setIsPlaying(false);
      setStatus("idle");
      return;
    }

    shouldReconnectRef.current = false;
    window.clearTimeout(reconnectTimerRef.current);
    socketRef.current?.close();
    setArenaStates([]);
    connectArena();
  }, [selectedAlgorithms, seed]);

  function toggleAlgorithm(id) {
    setSelectedAlgorithms((prev) => {
      if (prev.includes(id)) return prev.filter((algorithm) => algorithm !== id);
      if (prev.length >= MAX_INSTANCES) return prev;
      return [...prev, id];
    });
  }

  function connectArena() {
    if (selectedAlgorithms.length === 0) return;
    setStatus("connecting");
    setIsPlaying(false);
    setError("");
    shouldReconnectRef.current = false;
    window.clearTimeout(reconnectTimerRef.current);

    const socket = new WebSocket(arenaWebsocketUrl());
    socketRef.current = socket;

    socket.onopen = () => {
      if (socketRef.current !== socket) return;
      reconnectDelayRef.current = 1000;
      socket.send(JSON.stringify({ type: "init", algorithms: selectedAlgorithms, seed }));
      setStatus("running");
      setIsPlaying(true);
    };

    socket.onmessage = (event) => {
      if (socketRef.current !== socket) return;
      const payload = JSON.parse(event.data);
      if (payload.type === "pong") return;
      if (payload.type === "arena_error" || payload.error) {
        setError(payload.error);
        return;
      }
      if (payload.type === "arena_states") {
        setArenaStates(payload.states);
        if (payload.states.length > 0 && payload.states.every((state) => state.terminated || state.truncated)) {
          setIsPlaying(false);
        }
      }
    };

    socket.onerror = () => {
      if (socketRef.current !== socket) return;
      setError("WebSocket error");
      setIsPlaying(false);
    };

    socket.onclose = () => {
      if (socketRef.current !== socket) return;
      setIsPlaying(false);
      if (shouldReconnectRef.current) {
        const delay = reconnectDelayRef.current;
        reconnectDelayRef.current = Math.min(delay * 2, maxReconnectDelay);
        setStatus("connecting");
        reconnectTimerRef.current = window.setTimeout(connectArena, delay);
      } else {
        setStatus("idle");
      }
    };
  }

  function stepArena() {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "step" }));
    }
  }

  useEffect(() => {
    if (!isPlaying || status !== "running") return undefined;
    const intervalId = window.setInterval(() => {
      stepArena();
    }, Math.round(520 / speedMultiplier));
    return () => window.clearInterval(intervalId);
  }, [isPlaying, speedMultiplier, status]);

  function resetArena() {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "reset", seed }));
    }
  }

  const gridCount = Math.min(Math.max(arenaStates.length || selectedAlgorithms.length || 1, 1), 5);

  return (
    <section className="arena-section" id="arena">
      <div className="section-heading">
        <p className="section-kicker">{t("arena.title")}</p>
        <h2>{t("arena.title")}</h2>
        <p>{t("arena.description")}</p>
        <p className="section-note">{t("arena.limit_note")}</p>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="arena-controls">
        <fieldset className="arena-algo-select">
          <legend>{t("arena.select_algorithms")}</legend>
          <div className="arena-checkbox-grid">
            {algorithms.map((algorithm) => {
              const checked = selectedAlgorithms.includes(algorithm.id);
              const disabled = !algorithm.available || (!checked && selectedAlgorithms.length >= MAX_INSTANCES);
              const description = algorithm.available
                ? (algorithm.load_error || "Ready")
                : (algorithm.load_error || "Unavailable");
              return (
                <label
                  key={algorithm.id}
                  className={`arena-checkbox-label ${disabled ? "disabled" : ""}`}
                  title={description}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={disabled}
                    onChange={() => toggleAlgorithm(algorithm.id)}
                  />
                  <span className="arena-algo-label">
                    {algorithm.name}
                    {!algorithm.available ? <small>({description})</small> : null}
                  </span>
                </label>
              );
            })}
          </div>
        </fieldset>

        <div className="arena-actions">
          <label className="arena-seed-label">
            Seed (opsional)
            <select
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="arena-seed-input"
            >
              {MODEL_SEEDS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>

          <label className="arena-speed-label">
            <span className="arena-speed-label-row">
              <span>{t("control.speed")}</span>
              <strong>{speedMultiplier}x</strong>
            </span>
            <input
              type="range"
              min="1"
              max="5"
              step="1"
              value={speedMultiplier}
              onChange={(event) => setSpeedMultiplier(Number(event.target.value))}
            />
            <span className="arena-speed-hint">1x lebih lambat, 5x paling cepat</span>
          </label>

          {status === "running" ? (
            <>
              <button
                type="button"
                className={isPlaying ? "btn-pause" : "btn-play"}
                onClick={() => setIsPlaying((current) => !current)}
              >
                {isPlaying ? t("control.pause") : t("control.play")}
              </button>
              <button type="button" className="btn-step" onClick={stepArena} disabled={isPlaying}>
                {t("control.step")}
              </button>
              <button type="button" className="btn-reset" onClick={resetArena}>
                {t("control.reset")}
              </button>
            </>
          ) : (
            <span className="arena-autostart-hint">
              {status === "connecting" ? t("arena.status.connecting") : t("arena.status.idle")}
            </span>
          )}
        </div>
      </div>

      <div className={`arena-grid count-${gridCount}`}>
        {arenaStates.length === 0 && status === "idle" ? (
          <div className="arena-placeholder">
            {t("arena.no_selection")}
          </div>
        ) : null}

        {arenaStates.map((state) => (
          <article key={state.algorithm} className="arena-card">
            <header className="arena-card-header">
              <span className="arena-algo-badge">{state.algorithm.toUpperCase()}</span>
              <span className={`arena-status-badge ${state.terminated || state.truncated ? "status-done" : "status-running"}`}>
                {state.terminated || state.truncated
                  ? "DONE"
                  : isPlaying
                    ? `${t("arena.status.running")} • ${t("control.auto_policy")}`
                    : `${t("arena.status.running")} • ${t("control.step_action")}`}
              </span>
            </header>
            <div className="arena-canvas-wrap">
              <MiniGridCanvas state={state} />
            </div>
            <div className="arena-stats">
              <div className="arena-stat">
                <span>{t("arena.battery")}</span>
                <strong>{state.battery}/{state.battery_capacity}</strong>
              </div>
              <div className="arena-stat">
                <span>{t("arena.step")}</span>
                <strong>{state.step_count}</strong>
              </div>
              <div className="arena-stat">
                <span>{t("arena.reward")}</span>
                <strong>{state.episode_return?.toFixed(1)}</strong>
              </div>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}