import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { VaEditor } from "./VaEditor";
import "./va-editor.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <VaEditor />
  </StrictMode>,
);
