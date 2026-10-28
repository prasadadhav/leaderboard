import React, { useEffect, useRef, useState } from "react";
import { useTheme } from "../contexts/ThemeContext";

const UI_ACTION_PREFIX = "__UI_ACTION__";

interface ChatMessage {
  text: string;
  sender: "user" | "agent";
  isMarkdown?: boolean;
}

// Minimal inline markdown → HTML (bold, italic, inline-code, newlines only)
function renderMarkdown(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br />");
}

export const AgentComponent: React.FC = () => {
  const { colors } = useTheme();
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const agentUrl = import.meta.env.VITE_AGENT_URL || "ws://localhost:8765";

  const connect = () => {
    if (unmountedRef.current) return;
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;

    setIsConnecting(true);
    try {
      const ws = new WebSocket(agentUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (unmountedRef.current) { ws.close(); return; }
        setIsConnected(true);
        setIsConnecting(false);
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          let text = payload.message ?? JSON.stringify(payload);

          // Detect and dispatch UI action — never show in chat
          if (typeof text === "string" && text.startsWith(UI_ACTION_PREFIX)) {
            const json = text.slice(UI_ACTION_PREFIX.length);
            try {
              const action = JSON.parse(json);
              if (action.type === "apply_filters") {
                window.dispatchEvent(
                  new CustomEvent("letzeval:apply_filters", { detail: action.filters })
                );
              } else if (action.type === "reset_filters") {
                window.dispatchEvent(
                  new CustomEvent("letzeval:reset_filters", { detail: action })
                );
              }
            } catch {
              // malformed action — ignore silently
            }
            return;
          }

          setMessages((prev) => [
            ...prev,
            { text: String(text), sender: "agent", isMarkdown: true },
          ]);
        } catch {
          setMessages((prev) => [
            ...prev,
            { text: event.data, sender: "agent" },
          ]);
        }
      };

      ws.onerror = () => {
        setIsConnecting(false);
      };

      ws.onclose = () => {
        setIsConnected(false);
        setIsConnecting(false);
        // Auto-retry every 5 s while component is mounted
        if (!unmountedRef.current) {
          retryRef.current = setTimeout(connect, 5000);
        }
      };
    } catch {
      setIsConnecting(false);
      if (!unmountedRef.current) {
        retryRef.current = setTimeout(connect, 5000);
      }
    }
  };

  useEffect(() => {
    unmountedRef.current = false;
    connect();
    return () => {
      unmountedRef.current = true;
      if (retryRef.current) clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentUrl]);

  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      inputRef.current?.focus();
    }
  }, [messages, isOpen]);

  const sendMessage = () => {
    if (!input.trim() || !wsRef.current || !isConnected) return;
    setMessages((prev) => [...prev, { text: input, sender: "user" }]);
    wsRef.current.send(JSON.stringify({ action: "user_message", message: input }));
    setInput("");
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const dot = isConnected ? "#22c55e" : isConnecting ? "#f59e0b" : "#ef4444";

  // ── Floating launcher button ──────────────────────────────────────────────

  const launcherStyle: React.CSSProperties = {
    position: "fixed",
    bottom: 28,
    right: 28,
    zIndex: 1000,
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-end",
    gap: 12,
  };

  const fabStyle: React.CSSProperties = {
    width: 52,
    height: 52,
    borderRadius: "50%",
    backgroundColor: colors.accent,
    border: "none",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    boxShadow: "0 4px 16px rgba(0,0,0,0.25)",
    flexShrink: 0,
  };

  const panelStyle: React.CSSProperties = {
    width: 360,
    height: 480,
    borderRadius: 16,
    backgroundColor: colors.surface,
    border: `1px solid ${colors.border}`,
    boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  };

  const headerStyle: React.CSSProperties = {
    padding: "12px 16px",
    borderBottom: `1px solid ${colors.border}`,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: colors.surface,
    flexShrink: 0,
  };

  const msgAreaStyle: React.CSSProperties = {
    flex: 1,
    overflowY: "auto",
    padding: "12px 14px",
    display: "flex",
    flexDirection: "column",
    gap: 8,
  };

  const bubbleStyle = (sender: "user" | "agent"): React.CSSProperties => ({
    padding: "9px 13px",
    borderRadius: 12,
    maxWidth: "82%",
    wordWrap: "break-word",
    alignSelf: sender === "user" ? "flex-end" : "flex-start",
    backgroundColor: sender === "user" ? colors.accent : colors.pageBg,
    color: sender === "user" ? "#fff" : colors.textPrimary,
    fontSize: 13.5,
    lineHeight: 1.5,
  });

  const inputRowStyle: React.CSSProperties = {
    padding: "10px 12px",
    borderTop: `1px solid ${colors.border}`,
    display: "flex",
    gap: 8,
    flexShrink: 0,
    backgroundColor: colors.surface,
  };

  const inputStyle: React.CSSProperties = {
    flex: 1,
    padding: "8px 12px",
    border: `1px solid ${colors.border}`,
    borderRadius: 8,
    fontSize: 13.5,
    backgroundColor: colors.pageBg,
    color: colors.textPrimary,
    outline: "none",
  };

  const sendBtnStyle: React.CSSProperties = {
    padding: "8px 14px",
    backgroundColor: colors.accent,
    color: "#fff",
    border: "none",
    borderRadius: 8,
    cursor: isConnected ? "pointer" : "not-allowed",
    fontSize: 13.5,
    opacity: isConnected ? 1 : 0.55,
    flexShrink: 0,
  };

  return (
    <div style={launcherStyle}>
      {isOpen && (
        <div style={panelStyle}>
          {/* Header */}
          <div style={headerStyle}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  backgroundColor: dot,
                  display: "inline-block",
                }}
              />
              <span style={{ fontWeight: 600, fontSize: 14, color: colors.textPrimary }}>
                LëtzEval Assistant
              </span>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                color: colors.textMuted,
                fontSize: 18,
                lineHeight: 1,
                padding: 2,
              }}
              aria-label="Close chat"
            >
              ✕
            </button>
          </div>

          {/* Messages */}
          <div style={msgAreaStyle}>
            {messages.length === 0 && (
              <div
                style={{
                  textAlign: "center",
                  color: colors.textMuted,
                  fontSize: 13,
                  paddingTop: 24,
                }}
              >
                {isConnected
                  ? "Ask me about the leaderboard…"
                  : isConnecting
                  ? "Connecting…"
                  : "Agent offline"}
              </div>
            )}
            {messages.map((msg, i) =>
              msg.isMarkdown ? (
                <div
                  key={i}
                  style={bubbleStyle(msg.sender)}
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.text) }}
                />
              ) : (
                <div key={i} style={bubbleStyle(msg.sender)}>
                  {msg.text}
                </div>
              )
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input row */}
          <div style={inputRowStyle}>
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
              placeholder={isConnected ? "Type a message…" : "Not connected"}
              disabled={!isConnected}
              style={inputStyle}
            />
            <button onClick={sendMessage} disabled={!isConnected} style={sendBtnStyle}>
              Send
            </button>
          </div>
        </div>
      )}

      {/* FAB */}
      <button
        style={fabStyle}
        onClick={() => setIsOpen((o) => !o)}
        aria-label={isOpen ? "Close assistant" : "Open assistant"}
        title="LëtzEval Assistant"
      >
        {isOpen ? (
          // Down arrow / close indicator
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        ) : (
          // Chat bubble icon
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        )}
      </button>
    </div>
  );
};
