import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import './style.css';

type Evidence = { aid: string; text: string; law_id: string; title: string; score: number | null };
type Turn = { question: string; answer: string; evidence: Evidence[]; status: string; error: string };
type Event = { type: string; message?: string; text?: string; evidence?: Evidence[] };

function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState<boolean | null>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const check = () => fetch('/api/health').then(r => r.json()).then(d => setReady(d.ready)).catch(() => setReady(false));
    void check();
    const timer = setInterval(check, 15000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [turns]);

  async function send(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || busy) return;
    setQuestion(''); setBusy(true);
    setTurns(old => [...old, { question: text, answer: '', evidence: [], status: 'Đang kết nối…', error: '' }]);
    const update = (fn: (turn: Turn) => Turn) => setTurns(old => old.map((t, i) => i === old.length - 1 ? fn(t) : t));
    let completed = false;
    try {
      const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: text }) });
      if (!response.ok) {
        const body = await response.json();
        throw new Error(typeof body.detail === 'string' ? body.detail : 'Câu hỏi không hợp lệ. Vui lòng thử lại.');
      }
      if (!response.body) throw new Error('Không nhận được luồng trả lời.');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const accept = (line: string) => {
        if (!line.trim()) return;
        const item: Event = JSON.parse(line);
        if (item.type === 'error') throw new Error(item.message);
        if (item.type === 'done') completed = true;
        update(turn => ({ ...turn,
          status: item.type === 'status' ? item.message || '' : ['delta', 'done'].includes(item.type) ? '' : turn.status,
          answer: turn.answer + (item.type === 'delta' ? item.text || '' : ''),
          evidence: item.type === 'evidence' ? item.evidence || [] : turn.evidence,
        }));
      };
      try {
        while (true) {
          const { done, value } = await reader.read();
          buffer += decoder.decode(value, { stream: !done });
          const lines = buffer.split('\n'); buffer = lines.pop() || '';
          lines.forEach(accept);
          if (done) { if (buffer.trim()) accept(buffer); break; }
        }
      } finally { await reader.cancel(); reader.releaseLock(); }
      if (!completed) throw new Error('Kết nối bị gián đoạn. Vui lòng gửi lại câu hỏi.');
    } catch (error) {
      update(turn => ({ ...turn, status: '', error: error instanceof Error ? error.message : 'Chưa xử lý được câu hỏi.' }));
    } finally { setBusy(false); }
  }

  return <main>
    <header><div className="brand">⚖ <span>DRiLL · Tra cứu pháp luật</span></div><span className={`badge ${ready ? 'online' : ''}`} role="status">{ready === null ? 'Đang kết nối' : ready ? 'Sẵn sàng' : 'Chưa kết nối pipeline'}</span></header>
    <section className="intro"><p className="eyebrow">HỎI ĐÁP CÓ DẪN NGUỒN</p><h1>Tra cứu pháp luật Việt Nam</h1><p>Đặt câu hỏi, tìm căn cứ và đối chiếu nguyên văn các điều luật liên quan.</p></section>
    {!ready && ready !== null && <p className="notice">Pipeline chưa sẵn sàng. Nếu vừa khởi động notebook, hãy chờ nạp model; kiểm tra log Kaggle nếu trạng thái này kéo dài.</p>}
    <section aria-label="Lịch sử hỏi đáp">
      {turns.length === 0 && <div className="empty"><h2>Bắt đầu từ câu hỏi của bạn</h2><button className="example" onClick={() => setQuestion('Trường hợp nào được tạm hoãn gọi nhập ngũ?')}>Trường hợp nào được tạm hoãn gọi nhập ngũ? ↗</button><p>Câu trả lời dựa trên bằng chứng truy xuất, hỗ trợ tra cứu pháp luật.</p></div>}
      {turns.map((turn, i) => <article className="turn" key={i}>
        <div className="question"><span>BẠN</span><p>{turn.question}</p></div>
        <div className="answer"><span className="eyebrow">TRẢ LỜI</span><ReactMarkdown>{turn.answer}</ReactMarkdown>
          {turn.status && <p className="progress" role="status">{turn.status}</p>}
          {turn.error && <p className="error" role="alert">{turn.error}</p>}
          {turn.evidence.length > 0 && <details open><summary>Văn bản nguồn — trích nguyên văn ({turn.evidence.length} Điều)</summary>
            {turn.evidence.map((source, j) => <div className="source" key={`${source.aid}-${j}`}><h3>{j + 1}. aid {source.aid}{source.law_id && ` · ${source.law_id}`}{source.title && ` · ${source.title}`}</h3><p>{source.text}</p>{source.score !== null && <small>Điểm xếp hạng: {source.score.toFixed(4)}</small>}</div>)}
          </details>}
        </div>
      </article>)}<div ref={end}/>
    </section>
    <form onSubmit={send}><label htmlFor="question">Câu hỏi pháp luật</label><div className="composer"><textarea id="question" value={question} maxLength={10000} rows={2} placeholder="Nhập câu hỏi của bạn…" onChange={e => setQuestion(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); e.currentTarget.form?.requestSubmit(); } }}/><button disabled={busy || !question.trim()} type="submit">{busy ? 'Đang xử lý…' : 'Gửi câu hỏi ↑'}</button></div><small>Thông tin hỗ trợ tra cứu, không thay thế tư vấn pháp lý chuyên môn.</small></form>
  </main>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
