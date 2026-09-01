/**
 * Headless Chrome smoke test: open demo, wait for ready or error.
 * Run: node scripts/smoke-load.mjs
 */
import { chromium } from "playwright";

const URL = process.env.SMOKE_URL ?? "http://127.0.0.1:5173/";
const TIMEOUT_MS = Number(process.env.SMOKE_TIMEOUT_MS ?? 600_000);

const browser = await chromium.launch({
  headless: true,
  channel: "chrome",
});
const context = await browser.newContext();
const page = await context.newPage();

const logs = [];
page.on("console", (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));
page.on("pageerror", (err) => logs.push(`[pageerror] ${err.message}`));
page.on("crash", () => logs.push("[crash] page crashed"));

console.log(`Opening ${URL} (timeout ${TIMEOUT_MS / 1000}s)`);
await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 60_000 });

const sendBtn = page.locator(".composer button[type='submit']");
const deadline = Date.now() + TIMEOUT_MS;

while (Date.now() < deadline) {
  const disabled = await sendBtn.isDisabled();
  const pct = await page.locator(".load-bar-pct").textContent().catch(() => null);
  const err = await page.locator(".status-error").textContent().catch(() => null);
  if (err?.trim()) {
    console.error("ERROR:", err.trim());
    console.error(logs.slice(-20).join("\n"));
    await browser.close();
    process.exit(1);
  }
  if (!disabled) {
    console.log("READY — Send enabled", pct ? `(last ${pct})` : "");
    await sendBtn.click();
    await page.locator(".composer textarea").fill("hi");
    await sendBtn.click();
    await page.waitForTimeout(15_000);
    const transcript = await page.locator(".transcript").innerText();
    console.log("Transcript snippet:", transcript.slice(0, 300));
    await browser.close();
    process.exit(0);
  }
  process.stdout.write(`\rwaiting… ${pct ?? "—"}   `);
  try {
    await page.waitForTimeout(2000);
  } catch {
    console.error("\nPAGE CLOSED");
    console.error(logs.slice(-30).join("\n"));
    await browser.close();
    process.exit(1);
  }
}

console.error("\nTIMEOUT");
console.error(logs.slice(-30).join("\n"));
await browser.close();
process.exit(1);
