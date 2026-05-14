import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { ChatBox } from './ChatBox.jsx';

// jsdom doesn't implement scrollIntoView; stub it so the auto-scroll
// effect in ChatBox doesn't blow up.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

/** Build a fetch Response-like object whose body is a ReadableStream
 * yielding the given pre-formatted SSE event strings. */
function streamingResponse(eventStrings) {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      for (const evt of eventStrings) {
        controller.enqueue(encoder.encode(evt));
      }
      controller.close();
    },
  });
  return { ok: true, status: 200, body };
}

function sse(event, data) {
  return `event: ${event}\ndata: ${JSON.stringify({ phase: event, ...data })}\n\n`;
}

describe('ChatBox', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders an input and a Send button', () => {
    render(<ChatBox />);
    expect(screen.getByPlaceholderText(/ask a follow-up/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  it('disables Send when the input is empty', () => {
    render(<ChatBox />);
    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled();
  });

  it('streams the assistant answer token by token', async () => {
    global.fetch.mockResolvedValueOnce(
      streamingResponse([
        sse('evidence', { evidence: [] }),
        sse('token', { token: 'Einstein ' }),
        sse('token', { token: 'was ' }),
        sse('token', { token: 'born ' }),
        sse('token', { token: 'in Ulm.' }),
        sse('done', {}),
      ]),
    );

    render(<ChatBox />);
    const input = screen.getByPlaceholderText(/ask a follow-up/i);
    await userEvent.type(input, 'Where was Einstein born?');
    await userEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() => {
      expect(screen.getByText(/Einstein was born in Ulm\./)).toBeInTheDocument();
    });
    expect(screen.getByText('You')).toBeInTheDocument();
    expect(screen.getByText('Assistant')).toBeInTheDocument();
    expect(screen.getByText(/Where was Einstein born\?/)).toBeInTheDocument();
  });

  it('renders evidence from the evidence event', async () => {
    global.fetch.mockResolvedValueOnce(
      streamingResponse([
        sse('evidence', {
          evidence: [
            { chunk_index: 0, text: 'evidence text', score: 0.5 },
          ],
        }),
        sse('token', { token: 'A' }),
        sse('done', {}),
      ]),
    );

    render(<ChatBox />);
    await userEvent.type(
      screen.getByPlaceholderText(/ask a follow-up/i),
      'q',
    );
    await userEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() =>
      expect(screen.getByText(/^1 source from the article$/)).toBeInTheDocument(),
    );
  });

  it('keeps multi-turn history across requests', async () => {
    global.fetch.mockResolvedValueOnce(
      streamingResponse([
        sse('evidence', { evidence: [] }),
        sse('token', { token: 'first answer' }),
        sse('done', {}),
      ]),
    );
    global.fetch.mockResolvedValueOnce(
      streamingResponse([
        sse('evidence', { evidence: [] }),
        sse('token', { token: 'second answer' }),
        sse('done', {}),
      ]),
    );

    render(<ChatBox />);
    const input = screen.getByPlaceholderText(/ask a follow-up/i);

    await userEvent.type(input, 'first question');
    await userEvent.click(screen.getByRole('button', { name: /send/i }));
    await waitFor(() =>
      expect(screen.getByText(/first answer/)).toBeInTheDocument(),
    );

    await userEvent.type(input, 'second question');
    await userEvent.click(screen.getByRole('button', { name: /send/i }));
    await waitFor(() =>
      expect(screen.getByText(/second answer/)).toBeInTheDocument(),
    );

    expect(screen.getByText(/first question/)).toBeInTheDocument();
    expect(screen.getByText(/second question/)).toBeInTheDocument();

    // Second request should carry the first turn as history.
    expect(global.fetch).toHaveBeenCalledTimes(2);
    const secondBody = JSON.parse(global.fetch.mock.calls[1][1].body);
    expect(secondBody.history).toEqual([
      { role: 'user', text: 'first question' },
      { role: 'assistant', text: 'first answer' },
    ]);
  });

  it('clears the input after sending', async () => {
    global.fetch.mockResolvedValueOnce(
      streamingResponse([
        sse('evidence', { evidence: [] }),
        sse('token', { token: 'ok' }),
        sse('done', {}),
      ]),
    );

    render(<ChatBox />);
    const input = screen.getByPlaceholderText(/ask a follow-up/i);

    await userEvent.type(input, 'hello');
    expect(input).toHaveValue('hello');
    await userEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() => expect(input).toHaveValue(''));
  });

  it('shows the in-band error message and drops the empty assistant bubble', async () => {
    global.fetch.mockResolvedValueOnce(
      streamingResponse([
        sse('evidence', { evidence: [] }),
        sse('error', { error: 'service unavailable', type: 'LLMUnavailableError' }),
      ]),
    );

    render(<ChatBox />);
    await userEvent.type(
      screen.getByPlaceholderText(/ask a follow-up/i),
      'q',
    );
    await userEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() =>
      expect(screen.getByText(/service unavailable/)).toBeInTheDocument(),
    );
    // No empty assistant bubble should linger.
    expect(screen.queryByText('Assistant')).not.toBeInTheDocument();
  });

  it('shows transport error from a non-2xx response', async () => {
    global.fetch.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({ error: 'service unavailable' }),
    });

    render(<ChatBox />);
    await userEvent.type(
      screen.getByPlaceholderText(/ask a follow-up/i),
      'q',
    );
    await userEvent.click(screen.getByRole('button', { name: /send/i }));

    await waitFor(() =>
      expect(screen.getByText(/service unavailable/)).toBeInTheDocument(),
    );
  });

  it('does not submit when the input is whitespace-only', async () => {
    render(<ChatBox />);
    const sendButton = screen.getByRole('button', { name: /send/i });

    await userEvent.type(
      screen.getByPlaceholderText(/ask a follow-up/i),
      '   ',
    );
    expect(sendButton).toBeDisabled();
    expect(global.fetch).not.toHaveBeenCalled();
  });
});
