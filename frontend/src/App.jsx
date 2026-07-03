import React, { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { fetchAlgorithms, resetEpisode, websocketUrl } from "./api.js";
import ArenaSection from "./components/ArenaSection.jsx";
import ControlPanel from "./components/ControlPanel.jsx";
import GridCanvas from "./components/GridCanvas.jsx";
import InfoPanel from "./components/InfoPanel.jsx";

export default function App() {
  const { t, i18n } = useTranslation();
  const [algorithms, setAlgorithms] = useState([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("ppo");
  const [selectedAction, setSelectedAction] = useState("auto");
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [state, setState] = useState(null);
  const [error, setError] = useState("");
  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const reconnectDelayRef = useRef(1000);
  const maxReconnectDelay = 30000;
  const selectedAlgorithmRef = useRef(selectedAlgorithm);
  const shouldReconnectRef = useRef(true);
  const isManuallyClosedRef = useRef(false);

  const availableAlgorithms = useMemo(
    () => algorithms.filter((algorithm) => algorithm.available),
    [algorithms],
  );

  useEffect(() => {
    fetchAlgorithms()
      .then((payload) => {
        setAlgorithms(payload.algorithms);
        const availableAlgorithms = payload.algorithms.filter((algorithm) => algorithm.available);
        const preferred = availableAlgorithms.find((algorithm) => algorithm.id === "ppo") || availableAlgorithms[0];
        if (preferred) setSelectedAlgorithm(preferred.id);
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    selectedAlgorithmRef.current = selectedAlgorithm;
  }, [selectedAlgorithm]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    isManuallyClosedRef.current = false;
    reconnectDelayRef.current = 1000;
    connectSocket();
    return () => {
      shouldReconnectRef.current = false;
      isManuallyClosedRef.current = true;
      window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const intervalId = window.setInterval(() => {
      sendStep("auto");
    }, Math.round(520 / speedMultiplier));
    return () => window.clearInterval(intervalId);
  }, [isPlaying, speedMultiplier, selectedAlgorithm]);

  useEffect(() => {
    const heartbeatId = window.setInterval(() => {
      const socket = socketRef.current;
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "ping", algorithm: selectedAlgorithmRef.current }));
      }
    }, 5000);
    return () => window.clearInterval(heartbeatId);
  }, []);

  function sendStep(action = selectedAction) {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setIsPlaying(false);
      setError(t("error.websocket_disconnected"));
      setConnectionStatus("reconnecting");
      connectSocket();
      return;
    }
    socket.send(
      JSON.stringify({
        algorithm: selectedAlgorithm,
        action: action === "auto" ? null : action,
        mode: action === "auto" ? "auto" : "manual",
      }),
    );
  }

  function connectSocket() {
    const existing = socketRef.current;
    if (existing?.readyState === WebSocket.OPEN || existing?.readyState === WebSocket.CONNECTING) return;

    const socket = new WebSocket(websocketUrl("/ws/step"));
    socketRef.current = socket;
    setConnectionStatus((current) => (current === "closed" ? "reconnecting" : "connecting"));

    socket.onopen = () => {
      if (socketRef.current !== socket) return;
      reconnectDelayRef.current = 1000;
      setConnectionStatus("connected");
      setError("");
      socket.send(JSON.stringify({ type: "reset", algorithm: selectedAlgorithmRef.current }));
    };
    socket.onmessage = (event) => {
      if (socketRef.current !== socket) return;
      const payload = JSON.parse(event.data);
      if (payload.type === "pong") return;
      if (payload.error) {
        setError(payload.error);
        setIsPlaying(false);
        return;
      }
      setError("");
      setState(payload);
      if (payload.terminated || payload.truncated) {
        setIsPlaying(false);
      }
    };
    socket.onerror = () => {
      if (socketRef.current !== socket) return;
      setConnectionStatus("error");
      setIsPlaying(false);
    };
    socket.onclose = () => {
      if (socketRef.current !== socket) return;
      setConnectionStatus("closed");
      setIsPlaying(false);
      if (shouldReconnectRef.current) {
        const delay = reconnectDelayRef.current;
        setConnectionStatus("reconnecting");
        reconnectTimerRef.current = window.setTimeout(() => {
          reconnectDelayRef.current = Math.min(delay * 2, maxReconnectDelay);
          connectSocket();
        }, delay);
      }
    };
  }

  async function handleReset() {
    setIsPlaying(false);
    try {
      const restState = await resetEpisode(selectedAlgorithm);
      setState(restState);
      const socket = socketRef.current;
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "reset", algorithm: selectedAlgorithm }));
      }
    } catch (err) {
      setError(err.message);
    }
  }

  function handleAlgorithmChange(value) {
    setSelectedAlgorithm(value);
    setIsPlaying(false);
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "reset", algorithm: value }));
    }
  }

  function cycleLang() {
    i18n.changeLanguage(i18n.language === "en" ? "id" : "en");
  }

  return (
    <main className="app-shell">
      <header className="site-nav">
        <a className="brand-mark" href="#top" aria-label="Vacuum RL home">
          <span className="brand-symbol">VRL</span>
          <span>
            <strong>{t("brand.title")}</strong>
            <small>{t("brand.subtitle")}</small>
          </span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#demo">{t("nav.demo")}</a>
          <a href="#arena">{t("nav.arena")}</a>
          <a href="#about">{t("nav.about")}</a>
          <a href="https://github.com/cjoyy/vacuum-rl" target="_blank" rel="noreferrer">
            {t("nav.github")}
          </a>
          <button type="button" className="lang-switcher" onClick={cycleLang}>
            {i18n.language === "en" ? "EN" : "ID"}
          </button>
        </nav>
      </header>

      <section className="hero-section" id="top">
        <div className="hero-copy">
          <p className="section-kicker">{t("hero.kicker")}</p>
          <h1>{t("hero.title")}</h1>
          <p className="hero-subtitle">
            {t("hero.subtitle")}
          </p>
          <div className="hero-actions">
            <a className="primary-cta" href="#demo">{t("hero.cta.demo")}</a>
            <a className="secondary-cta" href="#about">{t("hero.cta.about")}</a>
          </div>
        </div>
        <div className="hero-panel" aria-label="Experiment overview">
          <div className="hero-panel-header">
            <span>{t("hero.panel.title")}</span>
            <strong className={`status-text status-${connectionStatus}`}>
              {connectionStatus === "connected" ? t("status.connected")
                : connectionStatus === "connecting" ? t("status.connecting")
                : connectionStatus === "reconnecting" ? t("status.reconnecting")
                : connectionStatus === "closed" ? t("status.closed")
                : connectionStatus === "error" ? t("status.error")
                : connectionStatus}
            </strong>
          </div>
          <div className="hero-panel-grid">
            <span>{t("hero.panel.algorithms")}</span>
            <strong>{availableAlgorithms.length || 5}</strong>
            <span>{t("hero.panel.actions")}</span>
            <strong>7</strong>
            <span>{t("hero.panel.environment")}</span>
            <strong>7 x 7</strong>
          </div>
        </div>
      </section>

      <section className="stats-section" aria-label="Experiment highlights">
        <HighlightCard value="71.67%" label={t("stat.success_rate")} detail={t("stat.success_rate_detail")} />
        <HighlightCard value="0.8956" label={t("stat.clean_cell_ratio")} detail={t("stat.clean_cell_ratio_detail")} />
        <HighlightCard value="5" label={t("stat.algorithms_compared")} detail={t("stat.algorithms_compared_detail")} />
      </section>

      <section className="demo-section" id="demo">
        <div className="section-heading">
          <p className="section-kicker">{t("demo.kicker")}</p>
          <h2>{t("demo.title")}</h2>
          <p>{t("demo.description")}</p>
          <p className="section-note">{t("demo.limit_note")}</p>
        </div>

        {error ? <div className="error-banner">{error}</div> : null}

        <div className="workspace">
          <GridCanvas state={state} />
          <aside className="demo-side">
            <ControlPanel
              algorithms={algorithms}
              selectedAlgorithm={selectedAlgorithm}
              selectedAction={selectedAction}
              speedMultiplier={speedMultiplier}
              isPlaying={isPlaying}
              connectionStatus={connectionStatus}
              canSend={connectionStatus === "connected"}
              onAlgorithmChange={handleAlgorithmChange}
              onActionChange={setSelectedAction}
              onSpeedChange={setSpeedMultiplier}
              onReset={handleReset}
              onStep={() => sendStep()}
              onPlayPause={() => setIsPlaying((current) => !current)}
            />
            <InfoPanel state={state} />
          </aside>
        </div>
      </section>

      <ArenaSection />

      <section className="about-section" id="about">
        <div className="section-heading">
          <p className="section-kicker">{t("about.kicker")}</p>
          <h2>{t("about.title")}</h2>
        </div>
        <div className="pdf-viewer">
          <object data="/paper.pdf" type="application/pdf" className="pdf-iframe" aria-label="Project paper">
            <p style={{ padding: 16 }}>
              PDF tidak bisa ditampilkan di browser ini. <a href="/paper.pdf" target="_blank" rel="noreferrer">Buka paper</a>.
            </p>
          </object>
        </div>
      </section>

      <footer className="site-footer">
        <div>
          <strong>{t("footer.title")}</strong>
          <p>{t("footer.description")}</p>
        </div>
        <div className="footer-links">
          <a href="https://github.com/cjoyy/vacuum-rl" target="_blank" rel="noreferrer">{t("footer.github")}</a>
          <span>{t("footer.author")}</span>
          <span>{t("footer.reference")}</span>
        </div>
      </footer>
    </main>
  );
}

function HighlightCard({ value, label, detail }) {
  return (
    <article className="highlight-card">
      <strong>{value}</strong>
      <span>{label}</span>
      <p>{detail}</p>
    </article>
  );
}
