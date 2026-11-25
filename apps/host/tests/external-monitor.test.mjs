import test from 'node:test';
import assert from 'node:assert/strict';
import * as monitor from '../server/monitor.mjs';

test('external engine metrics reach the monitor and are cleared on failure', async t => {
  const previousUrl = process.env.STRATA_SERVER_URL, previousFetch = globalThis.fetch;
  t.after(() => {
    if (previousUrl === undefined) delete process.env.STRATA_SERVER_URL;
    else process.env.STRATA_SERVER_URL = previousUrl;
    globalThis.fetch = previousFetch;
  });
  process.env.STRATA_SERVER_URL = 'http://127.0.0.1:8088/v1';
  let url;
  globalThis.fetch = async input => {
    url = input;
    return new Response(JSON.stringify({
      engine: { model: 'gemma', max_context: 131072 },
      live: { state: 'generating', prompt_tokens: 40, generated: 8, tok_s: 9, queued: 0 },
      requests: [], totals: { requests: 0 }, hardware: {}, history: {},
    }));
  };
  await monitor.pollExternalMetrics();
  assert.equal(url, 'http://127.0.0.1:8088/metrics?requests=all');
  assert.equal(monitor.snapshot().context.maxTokens, 131072);
  assert.equal(monitor.snapshot().outputTps, 9);
  assert.equal(monitor.snapshot().telemetry.throughput, 'engine');
  globalThis.fetch = async () => new Response(JSON.stringify({
    engine: { model: 'gemma', max_context: 131072 },
    live: { state: 'reading', prompt_tokens: 40, prompt_total: 40, prompt_read: null, prefill_tok_s_mean: null },
    requests: [], totals: { requests: 0 }, hardware: {}, history: {},
  }));
  await monitor.pollExternalMetrics();
  assert.equal(monitor.snapshot().prompt.done, null);
  assert.equal(monitor.snapshot().prompt.pct, null);
  assert.equal(monitor.snapshot().promptTps, null);
  globalThis.fetch = async () => new Response('', { status: 503 });
  await monitor.pollExternalMetrics();
  assert.notEqual(monitor.snapshot().telemetry.throughput, 'engine');
});
