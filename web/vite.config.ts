import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

const TRIALS_ROOT = path.resolve(__dirname, "../RAL/newmainexp");
const SAFE_NAME = /^[\w-]+$/;

/** Dev-only API for the VA trajectory editor (va-editor.html). */
function vaEditorApi(): Plugin {
  return {
    name: "va-editor-api",
    configureServer(server) {
      server.middlewares.use("/__va", (req, res) => {
        const url = new URL(req.url ?? "/", "http://localhost");
        const cond = url.searchParams.get("cond") ?? "emomanifold";
        const trial = url.searchParams.get("trial") ?? "004";
        const send = (status: number, body: unknown) => {
          res.statusCode = status;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify(body));
        };
        if (!SAFE_NAME.test(cond) || !SAFE_NAME.test(trial)) {
          send(400, { error: "bad cond/trial" });
          return;
        }
        const dir = path.join(TRIALS_ROOT, cond, trial);
        const original = path.join(dir, "playback_va.csv");
        const edited = path.join(dir, "playback_va_edited.csv");

        if (url.pathname === "/load" && req.method === "GET") {
          const wantOriginal = url.searchParams.has("original");
          const file = !wantOriginal && fs.existsSync(edited) ? edited : original;
          if (!fs.existsSync(file)) {
            send(404, { error: `not found: ${path.relative(TRIALS_ROOT, file)}` });
            return;
          }
          send(200, {
            source: path.relative(TRIALS_ROOT, file),
            text: fs.readFileSync(file, "utf-8"),
            originalText: fs.readFileSync(original, "utf-8"),
          });
          return;
        }

        if (url.pathname === "/save" && req.method === "POST") {
          const chunks: Buffer[] = [];
          req.on("data", (c: Buffer) => chunks.push(c));
          req.on("end", () => {
            fs.writeFileSync(edited, Buffer.concat(chunks).toString("utf-8"));
            send(200, { saved: path.relative(TRIALS_ROOT, edited) });
          });
          return;
        }

        send(404, { error: "unknown endpoint" });
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), vaEditorApi()],
  base: process.env.VITE_BASE || "./",
  resolve: {
    alias: {
      grimace: path.resolve(__dirname, "../grimace-main/src/index.ts"),
      "ts-deepmerge": path.resolve(__dirname, "node_modules/ts-deepmerge"),
    },
  },
  worker: {
    format: "es",
  },
});
