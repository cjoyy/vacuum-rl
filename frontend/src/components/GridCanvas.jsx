import React, { useEffect, useRef } from "react";

const COLORS = {
  background: "#f5f0e7",
  gridLine: "#d8ccbc",
  clean: "#fbf8f2",
  dirt1: "#f0c24b",
  dirt2: "#4bb6aa",
  dirt3: "#d96a5e",
  obstacle: "#3a3f46",
  dock: "#4b73b8",
  robot: "#2a2f36",
  robotCore: "#f8fbff",
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

    const backgroundGradient = context.createLinearGradient(0, 0, 0, rect.height);
    backgroundGradient.addColorStop(0, "#f7f1e8");
    backgroundGradient.addColorStop(1, COLORS.background);
    context.fillStyle = backgroundGradient;
    context.fillRect(0, 0, rect.width, rect.height);

    context.save();
    context.globalAlpha = 0.16;
    context.fillStyle = "#ffffff";
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const x = startX + col * cell;
        const y = startY + row * cell;
        context.fillRect(x, y, 1, cell);
        context.fillRect(x, y, cell, 1);
      }
    }
    context.restore();

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
        const cellGradient = context.createLinearGradient(x, y, x + size, y + size);

        if (isObstacle) {
          cellGradient.addColorStop(0, "#4b5159");
          cellGradient.addColorStop(1, COLORS.obstacle);
        } else if (isDock) {
          cellGradient.addColorStop(0, "#6f96d6");
          cellGradient.addColorStop(1, COLORS.dock);
        } else if (dirt === 0) {
          cellGradient.addColorStop(0, "#ffffff");
          cellGradient.addColorStop(1, COLORS.clean);
        } else if (dirt === 1) {
          cellGradient.addColorStop(0, "#ffe79b");
          cellGradient.addColorStop(1, COLORS.dirt1);
        } else if (dirt === 2) {
          cellGradient.addColorStop(0, "#93ddd7");
          cellGradient.addColorStop(1, COLORS.dirt2);
        } else {
          cellGradient.addColorStop(0, "#f1a09a");
          cellGradient.addColorStop(1, COLORS.dirt3);
        }

        context.fillStyle = cellGradient;
        context.beginPath();
        context.roundRect(x, y, size, size, 7);
        context.fill();
        context.strokeStyle = COLORS.gridLine;
        context.lineWidth = 1;
        context.stroke();

        if (isDock) {
          context.save();
          context.globalAlpha = 0.16;
          context.fillStyle = "#ffffff";
          context.beginPath();
          context.arc(x + size / 2, y + size / 2, size * 0.34, 0, Math.PI * 2);
          context.fill();
          context.restore();
        }

        if (!isObstacle && !isDock && dirt > 0) {
          context.save();
          context.globalAlpha = 0.18;
          context.fillStyle = "#fff8e8";
          context.beginPath();
          context.arc(x + size * 0.68, y + size * 0.3, Math.max(2.4, size * 0.08), 0, Math.PI * 2);
          context.fill();
          context.restore();
        }
      }
    }

    if (state.robot_pos) {
      const [robotRow, robotCol] = state.robot_pos;
      const x = startX + (robotCol - 1) * cell + cell / 2;
      const y = startY + (robotRow - 1) * cell + cell / 2;
      const radius = Math.max(12, cell * 0.25);
      const robotGradient = context.createRadialGradient(x - radius * 0.15, y - radius * 0.2, radius * 0.2, x, y, radius);
      robotGradient.addColorStop(0, "#454b54");
      robotGradient.addColorStop(1, COLORS.robot);
      context.fillStyle = robotGradient;
      context.save();
      context.shadowColor = "rgba(0, 0, 0, 0.2)";
      context.shadowBlur = 10;
      context.shadowOffsetY = 3;
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
      context.restore();

      context.strokeStyle = "rgba(255,255,255,0.14)";
      context.lineWidth = Math.max(1, cell * 0.03);
      context.beginPath();
      context.arc(x, y, radius - 1, 0, Math.PI * 2);
      context.stroke();

      context.fillStyle = COLORS.robotCore;
      context.beginPath();
      context.arc(x + radius * 0.25, y - radius * 0.2, radius * 0.32, 0, Math.PI * 2);
      context.fill();

      context.fillStyle = "rgba(255,255,255,0.12)";
      context.beginPath();
      context.arc(x - radius * 0.2, y - radius * 0.22, radius * 0.42, 0, Math.PI * 2);
      context.fill();
    }
  }, [state]);

  return (
    <div className="board-shell">
      <canvas ref={canvasRef} className="h-full w-full" aria-label="Vacuum grid simulation" />
      <div className="grid-legend">
        <span className="legend-item"><span className="swatch swatch-robot" /> Robot</span>
        <span className="legend-item"><span className="swatch swatch-dock" /> Dock</span>
        <span className="legend-item"><span className="swatch swatch-dirt1" /> Dirt lvl 1</span>
        <span className="legend-item"><span className="swatch swatch-dirt2" /> Dirt lvl 2</span>
        <span className="legend-item"><span className="swatch swatch-dirt3" /> Dirt lvl 3</span>
        <span className="legend-item"><span className="swatch swatch-obstacle" /> Obstacle</span>
      </div>
    </div>
  );
}
