const DEFAULT_API_URL = "http://localhost:8000";

export const API_URL = (import.meta.env.VITE_API_URL || DEFAULT_API_URL).replace(/\/$/, "");

export function websocketUrl(path) {
  const url = new URL(API_URL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = path;
  return url.toString();
}

export async function fetchAlgorithms() {
  const response = await fetch(`${API_URL}/algorithms`);
  if (!response.ok) {
    throw new Error(`Failed to load algorithms: ${response.status}`);
  }
  return response.json();
}

export async function resetEpisode(algorithm) {
  const response = await fetch(`${API_URL}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ algorithm }),
  });
  if (!response.ok) {
    throw new Error(`Failed to reset episode: ${response.status}`);
  }
  return response.json();
}
