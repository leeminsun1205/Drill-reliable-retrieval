import test from 'node:test';
import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import { contextFor, beforeEdit, exportMarkdown, linkCitations, loadConversations, newConversation, newTurn } from '../.test-build/state.js';
globalThis.crypto ??= webcrypto;
const completed = (q, a = 'Trả lời') => ({ ...newTurn(q), answer: a, complete: true, status: '' });

test('five latest successful turns only; errors and partials excluded', () => {
  const turns = Array.from({ length: 7 }, (_, i) => completed(`Q${i}`));
  turns.push(newTurn('pending'), { ...completed('failed'), error: 'offline' });
  const history = contextFor(turns);
  assert.deepEqual(history.map(t => t.question), ['Q2', 'Q3', 'Q4', 'Q5', 'Q6']);
  assert.equal(contextFor([completed('Q', 'x'.repeat(7000))])[0].answer.length, 6000);
});
test('editing or retrying uses only earlier context and cuts later turns', () => {
  const turns = [completed('Q1'), completed('Q2'), completed('Q3')];
  assert.deepEqual(beforeEdit(turns, turns[1].id), [turns[0]]);
  assert.deepEqual(beforeEdit(turns, turns[0].id), []);
});
test('round-trip local history, isolate conversations and restore interrupted streams', () => {
  const a = { ...newConversation(), turns: [completed('A')] };
  const b = { ...newConversation(), turns: [newTurn('B')] };
  const restored = loadConversations(JSON.stringify([a, b]));
  assert.equal(restored[0].turns[0].complete, true);
  assert.equal(restored[1].turns[0].complete, false);
  assert.ok(restored[1].turns[0].error);
  assert.equal(contextFor(restored[1].turns).length, 0);
  assert.deepEqual(loadConversations('{corrupt'), []);
  assert.deepEqual(loadConversations('{}'), []);
});
test('only retrieved citations become source links', () => {
  const evidence = [{ aid: '123', text: 'Điều luật' }];
  assert.equal(linkCitations('Căn cứ [aid=123], [aid=999]', evidence), 'Căn cứ [aid=123](#source-123), [aid=999]');
});
test('export includes full answer and original article text', () => {
  const c = newConversation(); c.title = 'Hội thoại';
  c.turns = [{ ...completed('Câu hỏi', 'Trả lời [aid=123]'), evidence: [{ aid: '123', text: 'Nguyên văn', law_id: 'Luật A' }] }];
  const markdown = exportMarkdown(c);
  assert.ok(markdown.includes('# Hội thoại'));
  assert.ok(markdown.includes('Trả lời [aid=123]'));
  assert.ok(markdown.includes('aid=123 · Luật A'));
  assert.ok(markdown.includes('Nguyên văn'));
});
