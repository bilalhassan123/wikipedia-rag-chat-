import { useEffect, useRef, useState } from 'react';

const API = '/api';
const HISTORY_TURNS = 6;

export function ChatBox() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const question = input;
    const history = messages.slice(-HISTORY_TURNS).map((m) => ({
      role: m.role,
      text: m.text,
    }));

    // Append user message + an empty streaming assistant message that
    // grows as tokens arrive.
    setMessages((prev) => [
      ...prev,
      { role: 'user', text: question },
      { role: 'assistant', text: '', evidence: [], streaming: true },
    ]);
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `Request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let inBandError = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let separatorIndex;
        while ((separatorIndex = buffer.indexOf('\n\n')) >= 0) {
          const block = buffer.slice(0, separatorIndex);
          buffer = buffer.slice(separatorIndex + 2);
          if (!block.trim()) continue;

          const evt = parseEvent(block);
          if (evt.event === 'evidence') {
            setMessages((prev) => updateLastAssistant(prev, (m) => ({
              ...m,
              evidence: evt.data?.evidence ?? [],
            })));
          } else if (evt.event === 'token') {
            const token = evt.data?.token ?? '';
            setMessages((prev) => updateLastAssistant(prev, (m) => ({
              ...m,
              text: m.text + token,
            })));
          } else if (evt.event === 'error') {
            inBandError = evt.data?.error || 'Streaming failed';
          }
          // 'done' just terminates the loop on the next read.
        }
      }

      // Mark streaming complete (drops the blinking cursor).
      setMessages((prev) => updateLastAssistant(prev, (m) => ({
        ...m,
        streaming: false,
      })));

      if (inBandError) {
        setError(inBandError);
        // Drop the empty assistant bubble if we never got any tokens.
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.role === 'assistant' && !last.text) {
            return prev.slice(0, -1);
          }
          return prev;
        });
      }
    } catch (err) {
      setError(err.message);
      // Roll back the empty streaming bubble on transport failure.
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === 'assistant' && !last.text) {
          return prev.slice(0, -1);
        }
        return prev;
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <section className="chat">
      <ol className="messages" data-testid="messages">
        {messages.map((m, i) => (
          <li key={i} className={`msg msg-${m.role}`}>
            <div className="msg-role">{m.role === 'user' ? 'You' : 'Assistant'}</div>
            <div className="msg-bubble">
              {m.text}
              {m.streaming && <span className="streaming-cursor">▍</span>}
            </div>
            {m.evidence && m.evidence.length > 0 && (
              <details className="evidence">
                <summary>
                  {m.evidence.length}{' '}
                  {m.evidence.length === 1 ? 'source' : 'sources'} from the article
                </summary>
                <ul>
                  {m.evidence.map((e, j) => (
                    <li key={j}>
                      <code>[{e.chunk_index}]</code> {e.text}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </li>
        ))}
        <li ref={messagesEndRef} aria-hidden="true" />
      </ol>

      {error && <div className="error">{error}</div>}

      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          placeholder="Ask a follow-up…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
          autoFocus
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? 'Streaming…' : 'Send'}
        </button>
      </form>
    </section>
  );
}

function updateLastAssistant(messages, updater) {
  if (messages.length === 0) return messages;
  const last = messages[messages.length - 1];
  if (last.role !== 'assistant') return messages;
  const updated = [...messages];
  updated[updated.length - 1] = updater(last);
  return updated;
}

function parseEvent(block) {
  const lines = block.split('\n');
  let eventName = 'message';
  let data = '';
  for (const line of lines) {
    if (line.startsWith('event: ')) {
      eventName = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      data += line.slice(6);
    }
  }
  let parsed = null;
  if (data) {
    try {
      parsed = JSON.parse(data);
    } catch {
      parsed = null;
    }
  }
  return { event: eventName, data: parsed };
}
