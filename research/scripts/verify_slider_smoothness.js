// DIAGNOSTIC ONLY -- verifies the manual-adjustment sliders (a) never call
// the backend while dragging and (b) actually redraw the live preview
// canvas, using a real Chrome instance via puppeteer-core. Not part of the
// app or its test suite; run manually against the dev servers.
//
//   node research/scripts/verify_slider_smoothness.js
//
// Expects the Vite dev server (with API proxy) reachable at APP_URL below,
// and a real 6-reference-image file present at IMAGE_PATH.
const puppeteer = require("puppeteer-core");
const path = require("path");

const APP_URL = process.env.APP_URL || "http://localhost:5174/";
const CHROME_PATH = process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const IMAGE_PATH = path.resolve(__dirname, "../inputs/7.jpeg");

async function main() {
  const browser = await puppeteer.launch({ executablePath: CHROME_PATH, headless: "new" });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  const resultRequests = [];
  page.on("request", (req) => {
    const url = req.url();
    if (url.includes("/api/jobs/") && (url.includes("/result") || url.includes("/results/"))) {
      resultRequests.push({ url, when: Date.now() });
    }
  });

  console.log("navigating to", APP_URL);
  await page.goto(APP_URL, { waitUntil: "networkidle0" });

  const fileInput = await page.waitForSelector('input[type="file"]', { timeout: 15000 });
  await fileInput.uploadFile(IMAGE_PATH);
  console.log("uploaded", IMAGE_PATH);

  // Wait for the "After · 4K" badge / result panel to appear (job complete).
  await page.waitForFunction(
    () => document.body.innerText.includes("4K Ready") || document.body.innerText.includes("photo enhanced"),
    { timeout: 180000 },
  );
  console.log("job completed, result panel visible");

  // Let the base preview canvas load/decode before starting the drag test.
  await page.waitForSelector("canvas", { timeout: 15000 });
  await new Promise((r) => setTimeout(r, 500));

  const beforeCount = resultRequests.length;
  const beforeCanvasSnapshot = await page.evaluate(() => {
    const c = document.querySelector("canvas");
    return c ? c.toDataURL().slice(0, 200) : null;
  });

  // Locate the Brightness slider specifically (aria-label="Brightness").
  const sliderHandle = await page.waitForSelector('input[aria-label="Brightness"]', { timeout: 10000 });

  const t0 = Date.now();
  // Simulate a rapid min -> max -> min drag by firing many 'input' events in
  // quick succession, exactly like a fast pointer drag would.
  await page.evaluate(() => {
    const el = document.querySelector('input[aria-label="Brightness"]');
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    const sequence = [];
    for (let v = -100; v <= 100; v += 4) sequence.push(v);
    for (let v = 100; v >= -100; v -= 4) sequence.push(v);
    window.__sliderSteps = sequence.length;
    for (const v of sequence) {
      setter.call(el, String(v));
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });
  const dragWallMs = Date.now() - t0;

  // Give rAF/settle timers a moment to flush, then check the canvas
  // actually changed (i.e. the preview really redrew, not just state).
  await new Promise((r) => setTimeout(r, 400));
  const afterCanvasSnapshot = await page.evaluate(() => {
    const c = document.querySelector("canvas");
    return c ? c.toDataURL().slice(0, 200) : null;
  });

  const afterCount = resultRequests.length;

  // Reset the slider back to 0 and confirm it's instant + canvas reverts.
  const resetT0 = Date.now();
  await page.evaluate(() => {
    const el = document.querySelector('input[aria-label="Brightness"]');
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(el, "0");
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await new Promise((r) => setTimeout(r, 350));
  const resetMs = Date.now() - resetT0;

  console.log("\n== RESULTS ==");
  console.log(`slider steps dispatched: ~100 (min->max->min)`);
  console.log(`wall time to dispatch all 'input' events synchronously: ${dragWallMs}ms`);
  console.log(`backend /result or /results/ requests BEFORE drag: ${beforeCount}`);
  console.log(`backend /result or /results/ requests DURING+AFTER drag: ${afterCount}`);
  console.log(`new backend requests caused by dragging: ${afterCount - beforeCount}`);
  console.log(`canvas pixel snapshot changed during drag: ${beforeCanvasSnapshot !== afterCanvasSnapshot}`);
  console.log(`reset-to-default wall time: ${resetMs}ms`);

  const noBackendCalls = (afterCount - beforeCount) === 0;
  const canvasUpdated = beforeCanvasSnapshot !== afterCanvasSnapshot;
  console.log(`\nPASS(no-backend-calls-while-dragging): ${noBackendCalls}`);
  console.log(`PASS(canvas-actually-redrew): ${canvasUpdated}`);

  await browser.close();
  if (!noBackendCalls || !canvasUpdated) process.exit(1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
