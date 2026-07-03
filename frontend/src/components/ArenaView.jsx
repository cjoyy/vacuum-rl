import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import { fetchAlgorithms, arenaWebsocketUrl } from "../api.js";
import MiniGridCanvas from "./MiniGridCanvas.jsx";

const MAX_INSTANCES = 4;

export default function ArenaView() {
  const { t, i18n } = useTranslation();
  const [algorithms, setAlgorithms] = useState([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState([]);
  const [arenaStates, setArenaStates] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [seed, setSeed] = useState(12345);
  const socketRef = useRef(null);
  const reconnectDelayRef = useRef(1000);
  const maxReconnectDelay = 30000;
  const shouldReconnectRef = useRef(false);
  const reconnectTimerRef = useRef(null);

  useEffect(() => {
    fetchAlgorithms()
      .then((payload) => setAlgorithms(payload.algorithms.filter((a) => a.available)))
      .catch((err) => setError(err.message));
    return () => {
      shouldReconnectRef.current = false;
      window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, []);

  function toggleAlgorithm(id) {
    setSelectedAlgorithms((prev) => {
      if (prev.includes(id)) return prev.filter((a) => a !== id);
      if (prev.length >= MAX_INSTANCES) return prev;
      return [...prev, id];
    });
  }

  function connectArena() {
    if (selectedAlgorithms.length === 0) return;
    setStatus("connecting");
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
      }
    };

    socket.onerror = () => {
      if (socketRef.current !== socket) return;
      setError("WebSocket error");
    };

    socket.onclose = () => {
      if (socketRef.current !== socket) return;
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

  function resetArena() {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "reset", seed }));
    }
  }

  function stopArena() {
    shouldReconnectRef.current = false;
    window.clearTimeout(reconnectTimerRef.current);
    socketRef.current?.close();
    setStatus("idle");
  }

  function cycleLang() {
    i18n.changeLanguage(i18n.language === "en" ? "id" : "en");
  }

  return (
    <main className="app-shell">
      <header className="site-nav">
        <Link className="brand-mark" to="/" aria-label="Vacuum RL home">
          <span className="brand-symbol">VRL</span>
          <span>
            <strong>{t("brand.title")}</strong>
            <small>{t("brand.subtitle")}</small>
          </span>
        </Link>
        <nav aria-label="Primary navigation">
          <Link to="/">{t("nav.demo")}</Link>
          <a href="#arena-top">{t("nav.arena")}</a>
          <button type="button" className="lang-switcher" onClick={cycleLang}>
            {i18n.language === "en" ? "ID" : "EN"}
          </button>
        </nav>
      </header>

      <section className="arena-section" id="arena-top">
        <div className="section-heading">
          <p className="section-kicker">{t("arena.title")}</p>
          <h2>{t("arena.title")}</h2>
          <p>{t("arena.description")}</p>
        </div>

        {error ? <div className="error-banner">{error}</div> : null}

        <div className="arena-controls">
          <fieldset className="arena-algo-select">
            <legend>{t("arena.select_algorithms")} ({t("arena.max_instances")})</legend>
            <div className="arena-checkbox-grid">
              {algorithms.map((algorithm) => {
                const checked = selectedAlgorithms.includes(algorithm.id);
                const disabled = !checked && selectedAlgorithms.length >= MAX_INSTANCES;
                return (
                  <label key={algorithm.id} className={`arena-checkbox-label ${disabled ? "disabled" : ""}`}>
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={disabled}
                      onChange={() => toggleAlgorithm(algorithm.id)}
                    />
                    {algorithm.name}
                  </label>
                );
              })}
            </div>
          </fieldset>

          <div className="arena-actions">
            <label className="arena-seed-label">
              Seed
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="arena-seed-input"
              />
            </label>

            {status === "running" ? (
              <>
                <button type="button" onClick={stepArena}>
                  {t("control.step")}
                </button>
                <button type="button" onClick={resetArena}>
                  {t("control.reset")}
                </button>
                <button type="button" className="btn-stop" onClick={stopArena}>
                  {t("arena.stop")}
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={connectArena}
                disabled={selectedAlgorithms.length === 0}
              >
                {status === "connecting" ? t("arena.status.connecting") : t("arena.start")}
              </button>
            )}
          </div>
        </div>

        <div className={`arena-grid cols-${Math.min(arenaStates.length || 1, 2)}`}>
          {arenaStates.length === 0 && status === "idle" ? (
            <div className="arena-placeholder">
              {t("arena.no_selection")}
            </div>
          ) : null}

          {arenaStates.map((state) => (
            <article key={state.algorithm} className="arena-card">
              <header className="arena-card-header">
                <span className="arena-algo-badge">{state.algorithm.toUpperCase()}</span>
                <span className="arena-status-badge">
                  {state.terminated || state.truncated ? "DONE" : t("arena.status.running")}
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
    </main>
  );
}
