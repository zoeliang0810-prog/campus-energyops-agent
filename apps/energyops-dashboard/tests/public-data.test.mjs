import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const dataUrl = new URL("../src/data.json", import.meta.url);

test("public dashboard data remains simulation-only and omits actual telemetry", async () => {
  const dashboard = JSON.parse(await readFile(dataUrl, "utf8"));
  const summary = dashboard.queries.dispatch_summary.rows[0];
  const schedule = dashboard.queries.dispatch_schedule.rows;

  assert.equal(summary.simulationOnly, true);
  assert.equal(summary.executable, false);
  assert.equal(schedule.length, 24);
  assert.ok(schedule.every((row) => row.actualLoadKw === null));
  assert.ok(schedule.every((row) => row.actualPvKw === null));
  assert.doesNotMatch(JSON.stringify(dashboard), /DEEPSEEK_API_KEY|Bearer\s+/i);
});
