import { useEffect, useMemo, useRef, useState } from "react";
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
    const socket = new WebSocket(websocketUrl("/ws/step"));
    socketRef.current = socket;
    setConnectionStatus("connecting");

    socket.onopen = () => {
      setConnectionStatus("connected");
      socket.send(JSON.stringify({ type: "reset", algorithm: selectedAlgorithm }));
    };
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
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
      setConnectionStatus("error");
      setIsPlaying(false);
    };
    socket.onclose = () => {
      setConnectionStatus("closed");
      setIsPlaying(false);
    };

    return () => socket.close();
  }, []);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const intervalId = window.setInterval(() => {
      sendStep("auto");
    }, speed);
    return () => window.clearInterval(intervalId);
  }, [isPlaying, speed, selectedAlgorithm]);

  function sendStep(action = selectedAction) {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(
      JSON.stringify({
        algorithm: selectedAlgorithm,
        action: action === "auto" ? null : action,
        mode: action === "auto" ? "auto" : "manual",
      }),
    );
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
      <header>
        <div>
          <p>vacuum-rl</p>
          <h1>Policy Rollout Console</h1>
        </div>
        <div className="api-chip">{availableAlgorithms.length} policies loaded</div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="workspace">
        <GridCanvas state={state} />
        <aside>
          <ControlPanel
            algorithms={algorithms}
            selectedAlgorithm={selectedAlgorithm}
            selectedAction={selectedAction}
            speed={speed}
            isPlaying={isPlaying}
            connectionStatus={connectionStatus}
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
    </main>
  );
}
