import MarkdownRenderer from "./MarkdownRenderer";

const RISK_CFG = {
  safe:    { label: "SAFE",    color: "var(--safe)",    bg: "rgba(0,255,136,.1)"  },
  caution: { label: "CAUTION", color: "var(--caution)", bg: "rgba(255,170,0,.1)"  },
  danger:  { label: "DANGER",  color: "var(--danger)",  bg: "rgba(255,51,85,.1)"  },
};

function RiskBadge({ risk }) {
  if (!risk || risk === "safe") return null;
  const cfg = RISK_CFG[risk];
  if (!cfg) return null;
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: "2px 8px",
      borderRadius: 10, border: `1px solid ${cfg.color}`,
      color: cfg.color, background: cfg.bg,
      letterSpacing: 1, textTransform: "uppercase",
      marginRight: 6,
    }}>
      {cfg.label}
    </span>
  );
}

export default function MessageBubble({ role, text, risk, planId, suggest, onConfirm, onSuggest, time }) {
  const isUser = role === "user";

  const bubbleStyle = {
    padding: "10px 14px",
    borderRadius: 10,
    fontSize: 13,
    lineHeight: 1.65,
    wordBreak: "break-word",
    border: "1px solid",
    ...(isUser
      ? {
          background: "rgba(0,212,255,.07)",
          borderColor: "var(--border2)",
          color: "var(--text)",
        }
      : {
          background: "var(--bg2)",
          borderColor: "var(--border)",
          color: "var(--text)",
        }
    ),
  };

  return (
    <div
      className={`msg-bubble ${role}`}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        gap: 4,
      }}
    >
      {/* Bubble */}
      <div style={bubbleStyle}>
        {isUser ? (
          <span className="selectable">{text}</span>
        ) : (
          <MarkdownRenderer text={text || ""} />
        )}

        {/* Confirm box — shown for danger risk with a plan */}
        {!isUser && risk === "danger" && planId && (
          <div className="confirm-box">
            <button
              className="confirm-btn exec"
              onClick={() => onConfirm && onConfirm(planId)}
            >
              EXECUTE
            </button>
            <button
              className="confirm-btn cancel"
              onClick={() => onConfirm && onConfirm(null)}
            >
              CANCEL
            </button>
          </div>
        )}

        {/* Suggest chip */}
        {!isUser && suggest && (
          <div style={{ marginTop: 10 }}>
            <button
              onClick={() => onSuggest && onSuggest(suggest)}
              className="quick-chip"
              style={{ fontSize: 11 }}
            >
              ↳ {suggest}
            </button>
          </div>
        )}
      </div>

      {/* Meta row: risk badge + timestamp */}
      <div style={{
        display: "flex", alignItems: "center",
        flexDirection: isUser ? "row-reverse" : "row",
        gap: 4,
        padding: "0 2px",
      }}>
        {risk && risk !== "safe" && <RiskBadge risk={risk} />}
        {time && (
          <span style={{ fontSize: 9, color: "var(--text3)" }}>
            {time}
          </span>
        )}
      </div>
    </div>
  );
}
