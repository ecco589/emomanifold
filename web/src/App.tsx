import { useEffect, useState } from "react";
import { copy, type Lang } from "./i18n";
import { Demo } from "./components/Demo";
import "./styles.css";

function parentLang(): Lang | null {
  try {
    if (window.parent === window) return null;
    const value = window.parent.document.documentElement.dataset.lang;
    return value === "zh" || value === "en" ? value : null;
  } catch {
    return null;
  }
}

function readLang(): Lang {
  const fromParent = parentLang();
  if (fromParent) return fromParent;
  const saved = localStorage.getItem("imcrl-lang") || localStorage.getItem("emomanifold-lang");
  if (saved === "zh" || saved === "en") return saved;
  return "en";
}

function Embed({ lang }: { lang: Lang }) {
  useEffect(() => {
    const report = () => {
      const root = document.getElementById("root");
      const height = Math.ceil(root ? root.getBoundingClientRect().bottom : 0);
      if (height > 0) window.parent.postMessage({ source: "emomanifold", height }, "*");
    };
    report();
    const observer = new ResizeObserver(report);
    observer.observe(document.body);
    return () => observer.disconnect();
  }, [lang]);

  return (
    <div className="page embed">
      <Demo lang={lang} />
    </div>
  );
}

export function App() {
  const [lang, setLang] = useState<Lang>(readLang);

  useEffect(() => {
    localStorage.setItem("emomanifold-lang", lang);
    document.documentElement.lang = lang === "zh" ? "zh-Hans" : "en";
  }, [lang]);

  useEffect(() => {
    const fromParent = parentLang();
    if (!fromParent) return;
    const el = window.parent.document.documentElement;
    const sync = () => {
      const value = el.dataset.lang;
      if (value === "zh" || value === "en") setLang(value);
    };
    const observer = new MutationObserver(sync);
    observer.observe(el, { attributes: true, attributeFilter: ["data-lang"] });
    return () => observer.disconnect();
  }, []);

  const embed = new URLSearchParams(window.location.search).has("embed");
  const t = copy[lang];

  if (embed) return <Embed lang={lang} />;

  return (
    <div className="page">
      <header className="top">
        <a className="brand" href="./">
          {t.brand}
        </a>
        <nav>
          <a href="https://github.com/ecco589/emomanifold">{t.links.github}</a>
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
