import { useEffect, useState } from "react";
import { copy, type Lang } from "./i18n";
import { Demo } from "./components/Demo";
import "./styles.css";

function readLang(): Lang {
  const saved = localStorage.getItem("emomanifold-lang");
  if (saved === "zh" || saved === "en") return saved;
  return "en";
}

export function App() {
  const [lang, setLang] = useState<Lang>(readLang);

  useEffect(() => {
    localStorage.setItem("emomanifold-lang", lang);
    document.documentElement.lang = lang === "zh" ? "zh-Hans" : "en";
  }, [lang]);

  const t = copy[lang];

  return (
    <div className="page">
      <header className="top">
        <a className="brand" href="./">
          {t.brand}
        </a>
        <nav>
          <a href="./">{t.links.github}</a>
          <a href="./">{t.links.arxiv}</a>
          <button
            type="button"
            className="lang"
            onClick={() => setLang(lang === "en" ? "zh" : "en")}
          >
            {t.langSwitch}
          </button>
        </nav>
      </header>

      <section className="hero">
        <p className="kicker">{t.brand}</p>
        <h1>{t.paperTitle}</h1>
        {t.authors ? <p className="authors">{t.authors}</p> : null}
      </section>

      <div className="notice">
        <p>{t.notice}</p>
        <p className="notice-local">{t.noticeLocal}</p>
      </div>

      <Demo lang={lang} />

      <section className="method">
        <div className="section-head">
          <h2>{t.methodTitle}</h2>
        </div>
        <figure>
          <img src="./fig/fig1.png" alt="EmoManifold system overview" />
          <figcaption>{t.methodCaption}</figcaption>
        </figure>
      </section>

      <section className="abstract">
        <h2>{t.abstractTitle}</h2>
        <p>{t.abstract}</p>
      </section>

      <footer>
        <p>{t.footer}</p>
      </footer>
    </div>
  );
}
