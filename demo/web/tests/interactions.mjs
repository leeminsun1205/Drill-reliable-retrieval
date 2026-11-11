// Real local HTTP streams exercise UI state; no external provider calls.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, extname } from 'node:path';
import { chromium } from 'playwright';
const root = fileURLToPath(new URL('../dist/', import.meta.url));
const open = new Set();
const evidence = { aid: '123', text: 'Nguồn kiểm tra giao diện.', law_id: 'TEST', title: '', score: 0.5 };
const server = createServer(async (req, res) => {
  if (req.url === '/api/health') { res.writeHead(200, { 'Content-Type': 'application/json' }).end('{"ready":true}'); return; }
  if (req.url === '/api/chat') {
    let raw = ''; for await (const chunk of req) raw += chunk;
    const { question } = JSON.parse(raw);
    open.add(res); res.on('close', () => open.delete(res));
    if (question === 'Chờ mãi') return; // no response headers
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson', 'Cache-Control': 'no-cache' });
    const event = data => res.write(JSON.stringify(data) + '\n');
    event({ type: 'evidence', evidence: [evidence] });
    event({ type: 'delta', text: 'Nội dung thử [aid=123].' });
    if (question === 'Đang chạy') { event({ type: 'status', message: 'Đang chờ dữ liệu…' }); return; }
    if (question === 'Nhiều chunk') {
      let count = 0;
      const timer = setInterval(() => {
        if (res.destroyed) { clearInterval(timer); return; }
        event({ type: 'delta', text: ' Từng phần.' });
        if (++count === 50) { clearInterval(timer); event({ type: 'done' }); res.end(); }
      }, 20);
      return;
    }
    event({ type: 'done' });
    if (question !== 'Done chưa đóng') res.end();
    return; // deliberately keep HTTP body open after done
  }
  const path = resolve(root, '.' + (req.url === '/' ? '/index.html' : (req.url || '').split('?')[0]));
  if (!path.startsWith(root)) { res.writeHead(403).end(); return; }
  try { res.writeHead(200, { 'Content-Type': ({ '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html' })[extname(path)] || 'application/octet-stream' }); res.end(await readFile(path)); }
  catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = []; page.on('pageerror', e => errors.push(e.message));
  await page.addInitScript(() => {
    window.storageWrites = 0;
    const write = Storage.prototype.setItem;
    Storage.prototype.setItem = function(...args) { window.storageWrites++; return write.apply(this, args); };
  });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const submit = async question => {
    await page.getByRole('textbox', { name: 'Câu hỏi pháp luật' }).fill(question);
    await page.getByRole('button', { name: 'Gửi ↑', exact: true }).click();
  };
  await submit('Done chưa đóng');
  await page.locator('.turn .answer').getByText('Nội dung thử', { exact: false }).waitFor();
  await page.getByRole('button', { name: 'Gửi ↑', exact: true }).waitFor({ timeout: 3000 });
  console.log('PASS: done releases UI even when HTTP connection stays open');
  await submit('Đang chạy');
  await page.getByText('Đang chờ dữ liệu…', { exact: false }).waitFor();
  await page.getByRole('button', { name: /^Đổi tên / }).first().click();
  await page.getByRole('textbox', { name: 'Tên hội thoại' }).fill('Hội thoại nền');
  await page.getByRole('button', { name: 'Lưu tên', exact: true }).click();
  await page.getByRole('button', { name: '＋ Cuộc trò chuyện mới' }).click();
  await page.getByRole('button', { name: 'Hội thoại nền', exact: true }).click();
  await page.getByRole('button', { name: 'aid=123', exact: true }).last().click();
  await page.getByRole('button', { name: 'Đóng nguồn', exact: true }).click();
  assert.equal(await page.locator('.source-panel').count(), 0);
  await page.getByRole('button', { name: 'Xóa Hội thoại nền', exact: true }).click();
  await page.keyboard.press('Escape');
  assert.equal(await page.getByRole('dialog').count(), 0);
  await page.getByRole('button', { name: 'Dừng trả lời', exact: true }).click();
  await page.getByRole('button', { name: 'Gửi ↑', exact: true }).waitFor();
  assert.ok(await page.getByText('Đã dừng lượt trả lời.', { exact: false }).count());
  console.log('PASS: navigation, rename, delete X and source close work during a stalled reply');
  await submit('Đang chạy');
  await page.getByText('Đang chờ dữ liệu…', { exact: false }).waitFor();
  await page.getByRole('button', { name: 'Xóa Hội thoại nền', exact: true }).click();
  await page.getByRole('button', { name: 'Xóa hội thoại', exact: true }).click();
  await page.getByRole('button', { name: 'Gửi ↑', exact: true }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Hội thoại nền', exact: true }).count(), 0);
  console.log('PASS: deleting the running conversation aborts it and unlocks submit');
  const writes = await page.evaluate(() => window.storageWrites);
  await submit('Nhiều chunk');
  await page.getByRole('button', { name: 'Gửi ↑', exact: true }).waitFor();
  const newWrites = await page.evaluate(() => window.storageWrites);
  assert.ok(newWrites - writes <= 4, `Unexpected synchronous storage writes: ${newWrites - writes}`);
  console.log('PASS: streaming does not serialize history on every chunk');
  await page.clock.install();
  await submit('Chờ mãi');
  await page.getByRole('button', { name: 'Dừng trả lời', exact: true }).waitFor();
  await page.clock.fastForward(91000);
  await page.getByRole('button', { name: 'Gửi ↑', exact: true }).waitFor({ timeout: 3000 });
  await page.getByText('Kết nối không nhận được dữ liệu trong 90 giây.', { exact: false }).waitFor();
  assert.deepEqual(errors, []);
  console.log('PASS: idle timeout unlocks controls when no headers/data arrive');
} finally {
  if (browser) await browser.close();
  for (const res of open) res.destroy();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
