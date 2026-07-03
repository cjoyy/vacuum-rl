import React, { useEffect, useRef } from "react";

const COLORS = {
  background: "#f7f4ee",
  gridLine: "#d7cec1",
  clean: "#fdfbf7",
  dirt1: "#e7c46f",
  dirt2: "#4ea3a8",
  dirt3: "#cf6f5f",
  obstacle: "#344054",
  dock: "#4169a8",
  robot: "#111827",
  robotCore: "#f8fafc",
};

function sameCell(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a[0] === b[0] && a[1] === b[1];
}

export default function GridCanvas({ state }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !state?.grid) return;

    const context = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    context.setTransform(dpr, 0, 0, dpr, 0, 0);

    const rows = state.grid.length;
    const cols = state.grid[0]?.length || 1;
    const gap = 3;
    const padding = 14;
    const cell = Math.min((rect.width - padding * 2) / cols, (rect.height - padding * 2) / rows);
    const boardWidth = cell * cols;
    const boardHeight = cell * rows;
    const startX = (rect.width - boardWidth) / 2;
    const startY = (rect.height - boardHeight) / 2;

    context.fillStyle = COLORS.background;
    context.fillRect(0, 0, rect.width, rect.height);

    const obstacleSet = new Set((state.obstacles || []).map(([row, col]) => `${row},${col}`));
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const envCell = [row + 1, col + 1];
        const x = startX + col * cell + gap;
        const y = startY + row * cell + gap;
        const size = cell - gap * 2;
        const dirt = state.grid[row][col];
        const isObstacle = obstacleSet.has(`${envCell[0]},${envCell[1]}`);
        const isDock = sameCell(envCell, state.dock);

        context.fillStyle = isObstacle
          ? COLORS.obstacle
          : isDock
            ? COLORS.dock
            : [COLORS.clean, COLORS.dirt1, COLORS.dirt2, COLORS.dirt3][dirt] || COLORS.clean;
        context.beginPath();
        context.roundRect(x, y, size, size, 7);
        context.fill();
        context.strokeStyle = COLORS.gridLine;
        context.lineWidth = 1;
        context.stroke();

        if (!isObstacle && !isDock && dirt > 0) {
          context.fillStyle = "rgba(255,255,255,0.72)";
          context.beginPath();
          context.arc(x + size * 0.68, y + size * 0.32, Math.max(3, size * 0.06), 0, Math.PI * 2);
          context.fill();
        }
      }
    }

    if (state.robot_pos) {
      const [robotRow, robotCol] = state.robot_pos;
      const x = startX + (robotCol - 1) * cell + cell / 2;
      const y = startY + (robotRow - 1) * cell + cell / 2;
      const radius = Math.max(12, cell * 0.25);
      context.fillStyle = COLORS.robot;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = COLORS.robotCore;
      context.beginPath();
      context.arc(x + radius * 0.25, y - radius * 0.2, radius * 0.32, 0, Math.PI * 2);
      context.fill();
    }
  }, [state]);

  return (
    <div className="board-shell">
      <canvas ref={canvasRef} className="h-full w-full" aria-label="Vacuum grid simulation" />
      <div className="grid-legend">
        <span className="legend-item"><span className="swatch swatch-robot" /> Robot</span>
        <span className="legend-item"><span className="swatch swatch-dock" /> Dock</span>
        <span className="legend-item"><span className="swatch swatch-dirt1" /> Dirt 1</span>
        <span className="legend-item"><span className="swatch swatch-dirt2" /> Dirt 2</span>
        <span className="legend-item"><span className="swatch swatch-dirt3" /> Dirt 3</span>
        <span className="legend-item"><span className="swatch swatch-obstacle" /> Obstacle</span>
      </div>
    </div>
  );
}
