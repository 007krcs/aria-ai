import { useStore } from "../store";

const COLORS = {
  info:    { bg: "rgba(74,158,255,.15)",  border: "#4a9eff", text: "#4a9eff" },
  success: { bg: "rgba(61,214,140,.15)",  border: "#3dd68c", text: "#3dd68c" },
  error:   { bg: "rgba(240,96,96,.15)",   border: "#f06060", text: "#f06060" },
  warning: { bg: "rgba(240,160,64,.15)",  border: "#f0a040", text: "#f0a040" },
};

export default function ToastContainer() {
  const { notifications } = useStore();
  if (!notifications.length) return null;

  return (
    <div style={{
      position: "fixed", bottom: 20, right: 20,
      display: "flex", flexDirection: "column", gap: 8,
      zIndex: 9999, pointerEvents: "none",
    }}>
      {notifications.map(n => {
        const c = COLORS[n.type] || COLORS.info;
        return (
          <div key={n.id} className="slide-in" style={{
            background: c.bg, border: `1px solid ${c.border}`,
            borderRadius: 8, padding: "10px 14px",
            fontSize: 12, color: c.text,
            maxWidth: 280, lineHeight: 1.5,
          }}>
            {n.msg}
          </div>
        );
      })}
    </div>
  );
}
