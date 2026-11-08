export type Evidence = { aid: string; text: string; law_id: string; title: string; score: number | null };
export type Turn = { id: string; question: string; answer: string; evidence: Evidence[]; suggestions: string[]; status: string; stage: string; error: string; warning: string; complete: boolean; elapsed: number; started: number };
export type Conversation = { id: string; title: string; turns: Turn[]; updated: number };
export const STORAGE_KEY = 'drill-legal-conversations-v1';
export const newConversation = (): Conversation => ({ id: crypto.randomUUID(), title: 'Cuộc trò chuyện mới', turns: [], updated: Date.now() });
export const newTurn = (question: string): Turn => ({ id: crypto.randomUUID(), question, answer: '', evidence: [], suggestions: [], status: 'Đang hiểu câu hỏi…', stage: 'understanding', error: '', warning: '', complete: false, elapsed: 0, started: Date.now() });
export function contextFor(turns: Turn[]) {
  return turns.filter(t => t.complete && !t.error && t.answer.trim()).slice(-5)
    .map(t => ({ question: t.question, answer: t.answer.slice(0, 6000) }));
}
export function beforeEdit(turns: Turn[], id: string) {
  const index = turns.findIndex(t => t.id === id);
  return index < 0 ? turns : turns.slice(0, index);
}
export function loadConversations(raw: string | null): Conversation[] {
  if (!raw) return [];
  try {
    const data: unknown = JSON.parse(raw);
    if (!Array.isArray(data)) return [];
    return data.filter((c): c is Conversation => !!c && typeof c.id === 'string' && typeof c.title === 'string' && Array.isArray(c.turns))
      .slice(0, 30).map(c => ({ ...c, turns: c.turns.filter(t => t && typeof t.question === 'string' && typeof t.answer === 'string' && typeof t.id === 'string' && Array.isArray(t.evidence))
        .map(t => ({ ...t, evidence: t.evidence.filter(e => e && typeof e.aid === 'string' && typeof e.text === 'string').map(e => ({ ...e, law_id: typeof e.law_id === 'string' ? e.law_id : '', title: typeof e.title === 'string' ? e.title : '', score: typeof e.score === 'number' && Number.isFinite(e.score) ? e.score : null })), suggestions: Array.isArray(t.suggestions) ? t.suggestions.filter(x => typeof x === 'string').slice(0, 3) : [],
          status: '', stage: '', error: t.complete ? t.error || '' : 'Lượt trả lời trước bị gián đoạn. Bạn có thể gửi lại.',
          complete: Boolean(t.complete), elapsed: Number(t.elapsed) || 0, warning: t.warning || '' })) }));
  } catch { return []; }
}
export function exportMarkdown(conversation: Conversation): string {
  return `# ${conversation.title}\n\n` + conversation.turns.map(t => `## Bạn\n\n${t.question}\n\n## Trả lời\n\n${t.answer}${t.error ? `\n\nLỗi: ${t.error}` : ''}`
    + (t.evidence.length ? '\n\n### Văn bản nguồn\n\n' + t.evidence.map(e => `#### aid=${e.aid}${e.law_id ? ` · ${e.law_id}` : ''}\n\n${e.text}`).join('\n\n') : '')).join('\n\n---\n\n');
}
export function linkCitations(text: string, evidence: Evidence[]) {
  const aids = new Set(evidence.map(e => e.aid));
  // Only IDs returned by retrieval get an interactive source target.
  return text.replace(/\[aid=([^\]\s]+)\](?!\()/g, (match, aid: string) => aids.has(aid) ? `[aid=${aid}](#source-${encodeURIComponent(aid)})` : match);
}
