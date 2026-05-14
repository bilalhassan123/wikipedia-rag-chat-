import { useState } from 'react';
import { ChatBox } from './ChatBox.jsx';

const API = '/api';

export default function App() {
  const [url, setUrl] = useState('');
  const [article, setArticle] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState([]);

  async function handleIndex(e) {
    e.preventDefault();
    setError(null);
    setIsLoading(true);
    setProgress([]);
    setArticle(null);

    try {
      const response = await fetch(`${API}/article/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        // Pre-stream error (e.g. Pydantic validation).
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `Request failed with status ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let resultData = null;
      let errorMessage = null;

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
          if (evt.event === 'error') {
            errorMessage = evt.data?.error || 'Something went wrong';
          } else if (evt.event === 'result') {
            resultData = evt.data;
          } else if (evt.data?.message) {
            setProgress((prev) => [...prev, evt.data.message]);
          }
        }
      }

      if (errorMessage) {
        setError(errorMessage);
      } else if (resultData) {
        setArticle(resultData);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main>
      <header>
        <h1>Article Companion</h1>
        <p className="hint">
          A local, source-grounded knowledge assistant for Wikipedia articles.
          Every answer cites the passages it was drawn from.
        </p>
      </header>

      <form onSubmit={handleIndex} className="url-form">
        <input
          type="url"
          placeholder="https://en.wikipedia.org/wiki/Albert_Einstein"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={isLoading}
          required
        />
        <button type="submit" disabled={isLoading || !url}>
          {isLoading ? 'Reading…' : 'Open article'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {(isLoading || (progress.length > 0 && !article)) && progress.length > 0 && (
        <ol className="progress" aria-live="polite">
          {progress.map((step, i) => {
            const isLast = i === progress.length - 1;
            const isCurrent = isLast && isLoading;
            return (
              <li
                key={i}
                className={`step ${isCurrent ? 'current' : 'done'}`}
              >
                {step}
              </li>
            );
          })}
        </ol>
      )}

      {article && (
        <>
          <section className="summary">
            <p className="article-label">Article — {article.title}</p>
            <h2>Summary</h2>
            <p className="summary-text">{article.summary}</p>
          </section>

          <section className="followups">
            <h2>Follow-up questions</h2>
            <p className="hint">
              Ask anything about the article. The model answers only from the
              retrieved passages and refuses if they don't support the question.
            </p>
            <ChatBox />
          </section>
        </>
      )}
    </main>
  );
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
