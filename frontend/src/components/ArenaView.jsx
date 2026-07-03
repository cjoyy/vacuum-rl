import React from "react";
import { useTranslation } from "react-i18next";
import { Link } from "react-router-dom";
import ArenaSection from "./ArenaSection.jsx";

export default function ArenaView() {
  const { t, i18n } = useTranslation();

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
          <a href="#arena">{t("nav.arena")}</a>
          <button type="button" className="lang-switcher" onClick={cycleLang}>
            {i18n.language === "en" ? "EN" : "ID"}
          </button>
        </nav>
      </header>

      <ArenaSection />
    </main>
  );
}
