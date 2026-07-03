import React, { Suspense } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./i18n";
import App from "./App.jsx";
import ArenaView from "./components/ArenaView.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Suspense fallback={<div style={{ padding: 40, textAlign: "center" }}>Loading...</div>}>
        <Routes>
          <Route path="/arena" element={<ArenaView />} />
          <Route path="*" element={<App />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  </React.StrictMode>,
);
