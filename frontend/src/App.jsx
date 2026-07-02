import React, { useEffect, useMemo, useRef, useState } from "react";
import { fetchAlgorithms, resetEpisode, websocketUrl } from "./api.js";
import ControlPanel from "./components/ControlPanel.jsx";
import GridCanvas from "./components/GridCanvas.jsx";
import InfoPanel from "./components/InfoPanel.jsx";

export default function App() {
  const [algorithms, setAlgorithms] = useState([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("ppo");
  const [selectedAction, setSelectedAction] = useState("auto");
  const [speed, setSpeed] = useState(520);
  const [isPlaying, setIsPlaying] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const [state, setState] = useState(null);
  const [error, setError] = useState("");
  const socketRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const selectedAlgorithmRef = useRef(selectedAlgorithm);
  const shouldReconnectRef = useRef(true);

  const availableAlgorithms = useMemo(
    () => algorithms.filter((algorithm) => algorithm.available),
    [algorithms],
  );

  useEffect(() => {
    fetchAlgorithms()
      .then((payload) => {
        setAlgorithms(payload.algorithms);
        const firstAvailable = payload.algorithms.find((algorithm) => algorithm.available);
        if (firstAvailable) setSelectedAlgorithm(firstAvailable.id);
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    selectedAlgorithmRef.current = selectedAlgorithm;
  }, [selectedAlgorithm]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connectSocket();
    return () => {
      shouldReconnectRef.current = false;
      window.clearTimeout(reconnectTimerRef.current);
      socketRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const intervalId = window.setInterval(() => {
      sendStep("auto");
    }, speed);
    return () => window.clearInterval(intervalId);
  }, [isPlaying, speed, selectedAlgorithm]);

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
      setError("WebSocket is disconnected. Reconnecting...");
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
        reconnectTimerRef.current = window.setTimeout(() => {
          setConnectionStatus("reconnecting");
          connectSocket();
        }, 1200);
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

  return (
    <main className="app-shell">
      <header className="site-nav">
        <a className="brand-mark" href="#top" aria-label="Vacuum RL home">
          <span className="brand-symbol">VRL</span>
          <span>
            <strong>Vacuum RL</strong>
            <small>Reinforcement Learning Simulator</small>
          </span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#demo">Demo</a>
          <a href="#about">About</a>
          <a href="https://github.com/cjoyy/vacuum-rl" target="_blank" rel="noreferrer">
            GitHub
          </a>
        </nav>
      </header>

      <section className="hero-section" id="top">
        <div className="hero-copy">
          <p className="section-kicker">Interactive research tool</p>
          <h1>Vacuum-cleaning robot dipelajari lewat 5 algoritma reinforcement learning</h1>
          <p className="hero-subtitle">
            Simulasi ini memvisualisasikan kebijakan DQN, TRPO, PPO, A2C, dan SAC pada
            lingkungan MDP vacuum-cleaning dengan battery constraint, obstacle, dan dinamika dirt.
          </p>
          <div className="hero-actions">
            <a className="primary-cta" href="#demo">Explore Demo</a>
            <a className="secondary-cta" href="#about">Read MDP Summary</a>
          </div>
        </div>
        <div className="hero-panel" aria-label="Experiment overview">
          <div className="hero-panel-header">
            <span>Policy rollout</span>
            <strong>{connectionStatus}</strong>
          </div>
          <div className="hero-panel-grid">
            <span>Algorithms</span>
            <strong>{availableAlgorithms.length || 5}</strong>
            <span>Actions</span>
            <strong>7</strong>
            <span>Environment</span>
            <strong>7 x 7</strong>
          </div>
        </div>
      </section>

      <section className="stats-section" aria-label="Experiment highlights">
        <HighlightCard value="71.67%" label="Success rate" detail="PPO policy quality evaluation" />
        <HighlightCard value="0.8956" label="Clean-cell ratio" detail="Average clean-state proportion" />
        <HighlightCard value="5" label="Algorithms compared" detail="DQN, TRPO, PPO, A2C, SAC" />
      </section>

      <section className="demo-section" id="demo">
        <div className="section-heading">
          <p className="section-kicker">Live simulator</p>
          <h2>Observe policy decisions step by step</h2>
          <p>
            Pilih algoritma, jalankan auto-policy, atau kirim action manual. Grid, battery,
            reward, dan episode return diperbarui melalui WebSocket.
          </p>
        </div>

        {error ? <div className="error-banner">{error}</div> : null}

        <div className="workspace">
          <GridCanvas state={state} />
          <aside className="demo-side">
            <ControlPanel
              algorithms={algorithms}
              selectedAlgorithm={selectedAlgorithm}
              selectedAction={selectedAction}
              speed={speed}
              isPlaying={isPlaying}
              connectionStatus={connectionStatus}
              canSend={connectionStatus === "connected"}
              onAlgorithmChange={handleAlgorithmChange}
              onActionChange={setSelectedAction}
              onSpeedChange={setSpeed}
              onReset={handleReset}
              onStep={() => sendStep()}
              onPlayPause={() => setIsPlaying((current) => !current)}
            />
            <InfoPanel state={state} />
          </aside>
        </div>
      </section>

      <section className="about-section" id="about">
        <div className="section-heading">
          <p className="section-kicker">MDP formulation</p>
          <h2>Vacuum cleaning as a constrained decision process</h2>
        </div>
        <div className="about-grid">
          <article>
            <h3>State</h3>
            <p>
              State mencakup posisi robot, level battery, status dirt pada local perception,
              obstacle, dan jarak relatif menuju charging dock.
            </p>
          </article>
          <article>
            <h3>Action</h3>
            <p>
              Agent memilih tujuh aksi diskrit: bergerak N/S/E/W, Stay, Clean, atau Charge.
              SAC dipetakan dari action kontinu ke action diskrit environment.
            </p>
          </article>
          <article>
            <h3>Reward</h3>
            <p>
              Reward menyeimbangkan pembersihan sel kotor, biaya langkah dan gerak, penalti bump,
              penalti action tidak efektif, charging behavior, dan reset saat battery habis.
            </p>
          </article>
        </div>
      </section>

      <footer className="site-footer">
        <div>
          <strong>Vacuum RL</strong>
          <p>Interactive companion for a reinforcement-learning vacuum-cleaner study.</p>
        </div>
        <div className="footer-links">
          <a href="https://github.com/cjoyy/vacuum-rl" target="_blank" rel="noreferrer">GitHub Repository</a>
          <span>Author/contributor: cjoyy</span>
          <span>Reference: project paper, Table 4 policy-quality results</span>
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
