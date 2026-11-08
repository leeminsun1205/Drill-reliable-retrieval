// UI integration with mocked API responses; no real models or provider calls.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, extname } from 'node:path';
import { chromium } from 'playwright';
const root = fileURLToPath(new URL('../dist/', import.meta.url));
const artifacts = fileURLToPath(new URL('../../../results/demo_ui_checks/', import.meta.url));
await mkdir(artifacts, { recursive: true });
const server = createServer(async (req, res) => {
  const path = resolve(root, '.' + (req.url === '/' ? '/index.html' : (req.url || '').split('?')[0]));
  if (!path.startsWith(root)) { res.writeHead(403).end(); return; }
  try {
    const data = await readFile(path);
    res.writeHead(200, { 'Content-Type': ({ '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html' })[extname(path)] || 'application/octet-stream' });
    res.end(data);
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, permissions: ['clipboard-read', 'clipboard-write'] });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  const requests = []; let failOnce = true;
  await page.route('**/api/health', route => route.fulfill({ json: { ready: true } }));
  await page.route('**/api/chat', async route => {
    const payload = route.request().postDataJSON(); requests.push(payload);
    if (payload.question === 'Thử lỗi' && failOnce) { failOnce = false; await route.fulfill({ status: 503, json: { detail: 'Dịch vụ tạm thời gián đoạn.' } }); return; }
    const events = payload.question === 'Vượt đèn đỏ phạt thế nào?' ? [
      { type: 'delta', text: 'Bạn hỏi về loại phương tiện nào?' },
      { type: 'suggestions', questions: ['Xe máy vượt đèn đỏ phạt thế nào?', 'Ô tô vượt đèn đỏ phạt thế nào?'] },
      { type: 'done' },
    ] : [
      { type: 'status', message: 'Đang tìm điều luật…', stage: 'retrieval' },
      { type: 'evidence', evidence: [{ aid: '123', law_id: 'Văn bản thử', title: 'Điều luật thử', text: 'Nguyên văn để kiểm tra giao diện. Đây không phải quy định pháp luật.', score: 0.5 }] },
      { type: 'delta', text: payload.question === 'Dài' ? ('Đoạn nội dung kiểm tra cuộn.\n\n'.repeat(100) + '[aid=123]') : 'Câu trả lời kiểm tra có nguồn [aid=123].' },
      { type: 'suggestions', questions: ['Tôi cần chuẩn bị giấy tờ gì?', 'Có thể thực hiện thủ tục ở đâu?', 'Tôi có quyền khiếu nại không?'] },
      { type: 'done' },
    ];
    await route.fulfill({ contentType: 'application/x-ndjson', body: events.map(e => JSON.stringify(e)).join('\n') + '\n' });
  });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.getByRole('heading', { name: /Bạn cần tìm hiểu/ }).waitFor();
  await page.screenshot({ path: resolve(artifacts, 'desktop-welcome.png') });
  const ask = async q => {
    await page.getByRole('textbox', { name: 'Câu hỏi pháp luật' }).fill(q);
    await page.getByRole('button', { name: 'Gửi ↑', exact: true }).click();
    await page.getByRole('button', { name: 'Gửi ↑', exact: true }).waitFor();
  };
  await ask('Vượt đèn đỏ phạt thế nào?');
  await page.getByText('Bạn hỏi về loại phương tiện nào?', { exact: true }).waitFor();
  await page.getByRole('button', { name: /Xe máy vượt đèn đỏ phạt thế nào/ }).click();
  await page.locator('.turn').nth(1).locator('.suggestions button').first().waitFor();
  assert.equal(requests[1].history.length, 1);
  assert.equal(await page.locator('.source-panel').count(), 0);
  await page.getByRole('button', { name: 'aid=123', exact: true }).click();
  await page.getByText('Nguyên văn để kiểm tra giao diện.', { exact: false }).waitFor();
  await page.screenshot({ path: resolve(artifacts, 'desktop-sources.png') });
  await page.getByRole('button', { name: 'Đóng nguồn', exact: true }).click();
  await page.getByRole('button', { name: 'Sao chép', exact: true }).last().click();
  assert.ok((await page.evaluate(() => navigator.clipboard.readText())).includes('[aid=123]'));
  await page.locator('.turn').last().locator('.suggestions button').first().click();
  await page.getByRole('button', { name: 'Gửi ↑', exact: true }).waitFor();
  assert.equal(requests.at(-1).question, 'Tôi cần chuẩn bị giấy tờ gì?');
  await page.getByRole('button', { name: 'Sửa câu hỏi', exact: true }).first().click();
  await ask('Tôi muốn hỏi về ô tô.');
  assert.equal(await page.locator('.turn').count(), 1);
  assert.deepEqual(requests.at(-1).history, []);
  await page.getByRole('button', { name: /^Đổi tên / }).first().click();
  await page.getByRole('textbox', { name: 'Tên hội thoại' }).fill('Hội thoại đã đổi tên');
  await page.getByRole('button', { name: 'Lưu tên', exact: true }).click();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Xuất .md ↗' }).click();
  const download = await downloadPromise;
  const markdown = await readFile(await download.path(), 'utf8');
  assert.ok(markdown.includes('Hội thoại đã đổi tên') && markdown.includes('aid=123'));
  await page.reload();
  await page.getByRole('button', { name: 'Hội thoại đã đổi tên', exact: true }).waitFor();
  assert.equal(await page.locator('.turn').count(), 1);
  assert.equal(await page.locator('.source-panel').count(), 0);
  await page.getByRole('button', { name: '＋ Cuộc trò chuyện mới' }).click();
  await ask('Thử lỗi');
  await page.getByRole('button', { name: 'Gửi lại', exact: true }).click();
  await page.getByRole('button', { name: 'Gửi ↑', exact: true }).waitFor();
  assert.deepEqual(requests.at(-1).history, []);
  assert.equal(await page.locator('.turn').count(), 1);
  await ask('Dài');
  await page.locator('.chat-scroll').evaluate(el => { el.scrollTop = 0; el.dispatchEvent(new Event('scroll')); });
  await page.getByRole('button', { name: '↓ Về tin nhắn mới nhất' }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('button', { name: 'Mở danh sách hội thoại' }).click();
  await page.getByRole('button', { name: 'Hội thoại đã đổi tên', exact: true }).click();
  await page.getByRole('button', { name: 'aid=123', exact: true }).click();
  await page.screenshot({ path: resolve(artifacts, 'mobile-sources.png') });
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('.source-panel').count(), 0);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.getByRole('button', { name: 'Mở danh sách hội thoại' }).click();
  await page.getByRole('button', { name: 'Xóa Hội thoại đã đổi tên', exact: true }).click();
  await page.getByRole('button', { name: 'Xóa hội thoại', exact: true }).click();
  assert.equal(await page.getByRole('button', { name: 'Hội thoại đã đổi tên', exact: true }).count(), 0);
  assert.deepEqual(errors, []);
  console.log('Browser checks passed: clarify, suggestions, citations, clipboard, edit, retry, rename, persist, export, delete, scroll and mobile layout.');
} catch (error) {
  const page = browser?.contexts()[0]?.pages()[0];
  if (page) await page.screenshot({ path: resolve(artifacts, 'failure.png') });
  throw error;
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
