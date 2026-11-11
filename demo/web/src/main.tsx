import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import { contextFor, beforeEdit, exportMarkdown, linkCitations, loadConversations, newConversation, newTurn, STORAGE_KEY } from './state';
import type { Conversation, Evidence, Turn } from './state';
import './style.css';

type StreamEvent = { type: string; message?: string; stage?: string; text?: string; evidence?: Evidence[]; questions?: string[] };
const examples = [
  ['Nghĩa vụ quân sự', 'Trường hợp nào được tạm hoãn gọi nhập ngũ?'],
  ['Lao động', 'Công ty nợ lương thì người lao động có thể làm gì?'],
  ['Đất đai', 'Cần chuẩn bị giấy tờ gì khi chuyển nhượng quyền sử dụng đất?'],
  ['Gia đình', 'Cha mẹ ly hôn thì quyền nuôi con được xác định thế nào?'],
];

type SourceSelection = { turnId: string; aid?: string } | null;
const AnswerBody = React.memo(function AnswerBody({ turn, openSource }: {
  turn: Turn; openSource: React.Dispatch<React.SetStateAction<SourceSelection>>;
}) {
  const components = useMemo(() => ({ a: ({ href, children }: React.ComponentProps<'a'>) => {
    if (href?.startsWith('#source-')) {
      let aid: string;
      try { aid = decodeURIComponent(href.slice(8)); } catch { return <span>{children}</span>; }
      if (turn.evidence.some(e => e.aid === aid)) return <button type="button" className="citation" onClick={() => openSource({ turnId: turn.id, aid })}>{children}</button>;
      return <span>{children}</span>;
    }
    return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
  } }), [turn.id, turn.evidence, openSource]);
  return <ReactMarkdown components={components}>{linkCitations(turn.answer, turn.evidence)}</ReactMarkdown>;
});

function App() {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    try { const saved = loadConversations(localStorage.getItem(STORAGE_KEY)); return saved.length ? saved : [newConversation()]; }
    catch { return [newConversation()]; }
  });
  const [activeId, setActiveId] = useState(conversations[0].id);
  const active = conversations.find(c => c.id === activeId) || conversations[0];
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const running = useRef<{ controller: AbortController; conversationId: string } | null>(null);
  const [ready, setReady] = useState<boolean | null>(null);
  const [notice, setNotice] = useState('');
  const [sidebar, setSidebar] = useState(false);
  const [source, setSource] = useState<SourceSelection>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [rename, setRename] = useState<string | null>(null);
  const [renameText, setRenameText] = useState('');
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const [clock, setClock] = useState(Date.now());
  const scroll = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const sourcePanel = useRef<HTMLElement>(null);
  const selectedTurn = active.turns.find(t => t.id === source?.turnId);

  const latestConversations = useRef(conversations);
  latestConversations.current = conversations;
  useEffect(() => {
    const save = () => {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(latestConversations.current)); }
      catch { setNotice('Không lưu được lịch sử trên trình duyệt này (bộ nhớ đầy hoặc đang bị chặn). Bạn có thể xuất hội thoại để giữ lại.'); }
    };
    if (!busy) { save(); return; }
    // Avoid synchronous serialization of all source text on every stream chunk.
    const timer = setTimeout(save, 750);
    return () => clearTimeout(timer);
  }, [conversations, busy]);
  useEffect(() => {
    const save = () => {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(latestConversations.current)); } catch { /* already reported by the normal save path */ }
    };
    window.addEventListener('pagehide', save);
    return () => window.removeEventListener('pagehide', save);
  }, []);
  useEffect(() => {
    const check = () => fetch('/api/health').then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(d => setReady(d.ready)).catch(() => setReady(false));
    void check(); const timer = setInterval(check, 15000); return () => clearInterval(timer);
  }, []);
  useEffect(() => { if (!busy) return; const timer = setInterval(() => setClock(Date.now()), 1000); return () => clearInterval(timer); }, [busy]);
  useEffect(() => { if (follow && scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight; }, [active.turns, follow]);
  useLayoutEffect(() => {
    const panel = sourcePanel.current;
    if (!panel) return;
    panel.scrollTop = 0;
    if (source?.aid && selectedTurn?.evidence[0]?.aid !== source.aid) {
      const article = document.getElementById(`source-${source.turnId}-${source.aid}`);
      if (article) panel.scrollTop = Math.max(0, article.getBoundingClientRect().top - panel.getBoundingClientRect().top - 20);
    }
  }, [source]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (deleteId) setDeleteId(null);
      else if (rename) setRename(null);
      else if (source) setSource(null);
      else if (sidebar) setSidebar(false);
    };
    document.addEventListener('keydown', close);
    return () => document.removeEventListener('keydown', close);
  }, [source, sidebar, rename, deleteId]);

  function patchConversation(id: string, fn: (c: Conversation) => Conversation) {
    setConversations(old => old.map(c => c.id === id ? fn(c) : c));
  }
  function choose(id: string) {
    setActiveId(id); setSource(null); setEditing(null); setQuestion(''); setSidebar(false); setFollow(true);
  }
  function createConversation() {
    if (conversations.length >= 30) { setNotice('Bạn đang có 30 cuộc trò chuyện. Hãy xuất và xóa một cuộc trò chuyện trước khi tạo mới.'); return; }
    const c = newConversation(); setConversations(old => [c, ...old]); setActiveId(c.id);
    setSource(null); setQuestion(''); setEditing(null); setSidebar(false); setFollow(true);
  }
  function removeConversation(id: string) {
    if (running.current?.conversationId === id) running.current.controller.abort();
    const remaining = conversations.filter(c => c.id !== id);
    if (!remaining.length) remaining.push(newConversation());
    setConversations(remaining); setDeleteId(null);
    if (active.id === id) { setActiveId(remaining[0].id); setQuestion(''); setSource(null); setEditing(null); }
  }
  function download() {
    const url = URL.createObjectURL(new Blob([exportMarkdown(active)], { type: 'text/markdown;charset=utf-8' }));
    const a = document.createElement('a'); a.href = url; a.download = `drill-${active.id}.md`; a.click(); URL.revokeObjectURL(url);
  }
  async function copy(text: string) {
    try { await navigator.clipboard.writeText(text); setNotice('Đã sao chép câu trả lời.'); }
    catch { setNotice('Trình duyệt chưa cho phép sao chép. Bạn có thể chọn văn bản để sao chép hoặc xuất hội thoại.'); }
  }
  async function ask(raw: string, replaceId?: string) {
    const text = raw.trim();
    if (!text || text.length > 10000 || inFlight.current) return;
    const controller = new AbortController();
    running.current = { controller, conversationId: active.id };
    let timedOut = false;
    let idleTimer: ReturnType<typeof setTimeout>;
    const watchIdle = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => { timedOut = true; controller.abort(); }, 90000);
    };
    watchIdle();
    inFlight.current = true; setBusy(true); setClock(Date.now());
    const conversationId = active.id;
    const base = replaceId ? beforeEdit(active.turns, replaceId) : active.turns;
    const history = contextFor(base);
    const turn = newTurn(text);
    const update = (fn: (turn: Turn) => Turn) => patchConversation(conversationId, c => ({ ...c, turns: c.turns.map(t => t.id === turn.id ? fn(t) : t) }));
    patchConversation(conversationId, c => ({ ...c, title: base.length === 0 && c.title === 'Cuộc trò chuyện mới' ? text.slice(0, 60) : c.title, turns: [...base, turn], updated: Date.now() }));
    setQuestion(''); setEditing(null); setSource(null); setFollow(true);
    let completed = false;
    try {
      const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text, history }), signal: controller.signal });
      watchIdle();
      if (!response.ok) {
        const body = await response.json();
        throw new Error(typeof body.detail === 'string' ? body.detail : 'Câu hỏi hoặc lịch sử không hợp lệ. Vui lòng rút ngắn và thử lại.');
      }
      if (!response.body) throw new Error('Không nhận được luồng trả lời.');
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
      const accept = (line: string) => {
        if (!line.trim()) return;
        const item: StreamEvent = JSON.parse(line);
        if (item.type === 'error') throw new Error(item.message);
        if (item.type === 'done') completed = true;
        update(t => ({ ...t,
          status: item.type === 'status' ? item.message || '' : ['delta', 'done'].includes(item.type) ? '' : t.status,
          stage: item.type === 'status' ? item.stage || t.stage : t.stage,
          answer: t.answer + (item.type === 'delta' ? item.text || '' : ''),
          evidence: item.type === 'evidence' ? item.evidence || [] : t.evidence,
          suggestions: item.type === 'suggestions' ? item.questions || [] : t.suggestions,
          warning: item.type === 'warning' ? item.message || '' : t.warning,
          complete: item.type === 'done' || t.complete,
          elapsed: Date.now() - turn.started,
        }));
      };
      try {
        while (true) {
          const { done, value } = await reader.read();
          watchIdle(); buffer += decoder.decode(value, { stream: !done });
          const lines = buffer.split('\n'); buffer = lines.pop() || '';
          for (const line of lines) { accept(line); if (completed) break; }
          if (completed) break;
          if (done) { if (buffer.trim()) accept(buffer); break; }
        }
      } finally {
        void reader.cancel().catch(() => {});
        reader.releaseLock();
      }
      if (!completed) throw new Error('Kết nối bị gián đoạn. Bạn có thể gửi lại câu hỏi.');
    } catch (error) {
      update(t => ({ ...t, status: '', complete: false, error: controller.signal.aborted ? (timedOut ? 'Kết nối không nhận được dữ liệu trong 90 giây. Bạn có thể gửi lại câu hỏi.' : 'Đã dừng lượt trả lời. Bạn có thể gửi lại khi cần.') : error instanceof Error ? error.message : 'Chưa xử lý được câu hỏi.', elapsed: Date.now() - turn.started }));
    } finally {
      clearTimeout(idleTimer!); controller.abort();
      if (running.current?.controller === controller) {
        running.current = null; inFlight.current = false; setBusy(false);
      }
    }
  }
  function startEdit(turn: Turn) {
    setEditing(turn.id); setQuestion(turn.question); input.current?.focus();
  }

  return <div className="app-shell">
    {sidebar && <button className="sidebar-shade" aria-label="Đóng danh sách hội thoại" onClick={() => setSidebar(false)}/>}
    <aside className={`sidebar ${sidebar ? 'shown' : ''}`} aria-label="Danh sách hội thoại">
      <div className="brand">⚖ <span>DRiLL<span className="brand-sub">PHÁP LUẬT VIỆT NAM</span></span></div>
      <button className="new-chat" onClick={createConversation}>＋ Cuộc trò chuyện mới</button>
      <p className="eyebrow sidebar-label">HỘI THOẠI CỦA BẠN</p>
      <nav>{conversations.map(c => <div className={`conversation ${c.id === active.id ? 'active' : ''}`} key={c.id}>
        <button className="conversation-title" title={c.title} onClick={() => choose(c.id)}>{c.title}</button>
        <button className="icon-button" aria-label={`Đổi tên ${c.title}`} onClick={() => { setRename(c.id); setRenameText(c.title); }}>✎</button>
        <button className="icon-button" aria-label={`Xóa ${c.title}`} onClick={() => setDeleteId(c.id)}>×</button>
      </div>)}</nav>
      <div className="sidebar-footer"><p>Lịch sử lưu trên trình duyệt này.</p><p>Không chia sẻ thiết bị nếu hội thoại chứa thông tin riêng tư.</p></div>
    </aside>
    <main className="workspace">
      <header><div className="header-title"><button className="menu-button" aria-label="Mở danh sách hội thoại" onClick={() => setSidebar(true)}>☰</button><span>{active.title}</span></div><div className="header-actions"><span className={`badge ${ready ? 'online' : ''}`} role="status">{ready === null ? 'Đang kết nối' : ready ? 'Sẵn sàng' : 'Pipeline chưa sẵn sàng'}</span><button className="text-button" onClick={download} disabled={!active.turns.length}>Xuất .md ↗</button></div></header>
      {notice && <div className="notice" role="status">{notice}<button aria-label="Đóng thông báo" onClick={() => setNotice('')}>×</button></div>}
      {!ready && ready !== null && <div className="notice">Đang chờ pipeline. Nếu vừa mở notebook, việc nạp model có thể mất vài phút. Bạn vẫn có thể hỏi về chức năng trợ lý.</div>}
      <div className="chat-scroll" ref={scroll} onScroll={() => { const el = scroll.current; if (el) setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 100); }}>
        <div className="chat-content">
          {!active.turns.length && <section className="welcome"><div className="welcome-icon">⚖</div><p className="eyebrow">TRA CỨU CÓ CĂN CỨ</p><h1>Bạn cần tìm hiểu<br/>vấn đề pháp luật nào?</h1><p>Đặt câu hỏi theo tình huống của bạn.<br/>Mỗi câu trả lời đi cùng nguồn để bạn đối chiếu.</p><div className="example-grid">{examples.map(([label, text]) => <button key={label} disabled={busy} onClick={() => void ask(text)}><span>{label} ↗</span>{text}</button>)}</div><small>Câu hỏi mẫu giúp bạn bắt đầu; phạm vi trả lời phụ thuộc bằng chứng tìm được.</small></section>}
          {active.turns.map(turn => <article className="turn" key={turn.id}>
            <div className="question"><span>BẠN</span><p>{turn.question}</p><button className="text-button" onClick={() => startEdit(turn)}>Sửa câu hỏi</button></div>
            <div className="answer"><p className="eyebrow">DRiLL · TRẢ LỜI</p><AnswerBody turn={turn} openSource={setSource}/>
              {turn.status && <div className="progress" role="status"><div className="steps">{[['understanding', 'Hiểu câu hỏi'], ['retrieval', 'Tìm điều luật'], ['answer', 'Soạn trả lời']].map(([key, label]) => <span className={turn.stage === key ? 'current' : ''} key={key}>{label}</span>)}</div><p><span className="spinner"/>{turn.status} <small>{Math.max(0, Math.floor((clock - turn.started) / 1000))} giây</small></p></div>}
              {turn.error && <p className="error" role="alert">{turn.error}</p>}
              {turn.warning && <p className="muted">{turn.warning}</p>}
              {(turn.answer || turn.error) && <div className="answer-actions">{turn.answer && <button className="text-button" onClick={() => void copy(turn.answer)}>Sao chép</button>}{turn.error && <button className="text-button" disabled={busy} onClick={() => void ask(turn.question, turn.id)}>Gửi lại</button>}{turn.evidence.length > 0 && <button className="source-toggle" aria-expanded={source?.turnId === turn.id} onClick={() => setSource(source?.turnId === turn.id ? null : { turnId: turn.id })}>▤ Văn bản nguồn · {turn.evidence.length} Điều</button>}{turn.complete && <small>{(turn.elapsed / 1000).toFixed(1)} giây</small>}</div>}
              {turn.complete && turn.suggestions.length > 0 && <div className="suggestions"><p className="eyebrow">BẠN CÓ THỂ HỎI TIẾP</p>{turn.suggestions.map(q => <button disabled={busy} key={q} onClick={() => void ask(q)}>{q}<span>↗</span></button>)}</div>}
            </div>
          </article>)}
        </div>
      </div>
      {!follow && <button className="jump-button" onClick={() => { setFollow(true); if (scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight; }}>↓ Về tin nhắn mới nhất</button>}
      <div className="composer-area">{busy && running.current?.conversationId !== active.id && <p className="muted">Đang trả lời hội thoại khác. Bạn có thể xem lịch sử hoặc bấm Dừng để gửi câu hỏi mới.</p>}{editing && <div className="editing">Đang sửa câu hỏi. Khi gửi, các lượt phía sau sẽ được bỏ để cập nhật ngữ cảnh.<button onClick={() => { setEditing(null); setQuestion(''); }}>Hủy</button></div>}
        <form onSubmit={e => { e.preventDefault(); void ask(question, editing || undefined); }}><label className="sr-only" htmlFor="question">Câu hỏi pháp luật</label><div className="composer"><textarea ref={input} id="question" value={question} maxLength={10000} rows={2} placeholder="Nhập câu hỏi pháp luật của bạn…" onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }}/><button className="stop-button" type="button" aria-label="Dừng trả lời" onClick={() => running.current?.controller.abort()} hidden={!busy}>Dừng ■</button>{!busy && <button disabled={!question.trim()} type="submit">Gửi ↑</button>}</div></form><small>Nhớ 5 lượt gần nhất · Thông tin hỗ trợ tra cứu, không thay thế tư vấn pháp lý chuyên môn.</small>
      </div>
    </main>
    {selectedTurn && <><button className="source-shade" aria-label="Đóng văn bản nguồn" onClick={() => setSource(null)}/><aside ref={sourcePanel} className="source-panel" aria-label="Văn bản nguồn"><div className="source-heading"><div><p className="eyebrow">ĐỐI CHIẾU BẰNG CHỨNG</p><h2>Văn bản nguồn</h2></div><button className="icon-button" autoFocus aria-label="Đóng nguồn" onClick={() => setSource(null)}>×</button></div><p className="muted">Nguyên văn từ corpus. Nội dung hiển thị không tự xác nhận hiệu lực hiện tại của văn bản.</p>{selectedTurn.evidence.map((e, i) => <section className={`source ${e.aid === source?.aid ? 'highlight' : ''}`} id={`source-${selectedTurn.id}-${e.aid}`} key={`${e.aid}-${i}`}><h3 className="source-title"><span>{i + 1}. aid {e.aid}</span>{e.score !== null && <span className="source-score">Điểm: {e.score.toFixed(4)}</span>}</h3>{e.law_id && <p className="law-id">{e.law_id}</p>}{e.title && <h4>{e.title}</h4>}<p className="source-text">{e.text}</p></section>)}</aside></>}
    {rename && <div className="modal-shade" onClick={e => { if (e.target === e.currentTarget) setRename(null); }}><form className="modal" role="dialog" aria-modal="true" aria-label="Đổi tên hội thoại" onSubmit={e => { e.preventDefault(); if (renameText.trim()) { patchConversation(rename, c => ({ ...c, title: renameText.trim() })); setRename(null); } }}><div className="modal-title"><h2>Đổi tên hội thoại</h2><button type="button" aria-label="Đóng đổi tên" onClick={() => setRename(null)}>×</button></div><input aria-label="Tên hội thoại" autoFocus maxLength={100} value={renameText} onChange={e => setRenameText(e.target.value)}/><div><button type="button" onClick={() => setRename(null)}>Hủy</button><button type="submit" disabled={!renameText.trim()}>Lưu tên</button></div></form></div>}
    {deleteId && <div className="modal-shade" onClick={e => { if (e.target === e.currentTarget) setDeleteId(null); }}><div className="modal" role="dialog" aria-modal="true" aria-label="Xóa hội thoại"><div className="modal-title"><h2>Xóa hội thoại này?</h2><button type="button" aria-label="Đóng xác nhận xóa" onClick={() => setDeleteId(null)}>×</button></div><p>Hội thoại sẽ bị xóa khỏi trình duyệt. Nếu đang trả lời, lượt đó sẽ được dừng. Bạn có thể xuất Markdown trước khi xóa.</p><div><button onClick={() => setDeleteId(null)}>Giữ lại</button><button className="danger" onClick={() => removeConversation(deleteId)}>Xóa hội thoại</button></div></div></div>}
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
