import { useEffect, useRef } from "react";

export default function NeuralCanvas({ height = 140 }) {
  const ref = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    let W = 0, H = 0;

    const NODE_COUNT = 11;
    const nodes = [];

    function initNodes() {
      nodes.length = 0;
      for (let i = 0; i < NODE_COUNT; i++) {
        nodes.push({
          x:  Math.random() * W,
          y:  Math.random() * H,
          vx: (Math.random() - 0.5) * 0.6,
          vy: (Math.random() - 0.5) * 0.6,
          r:  2 + Math.random() * 2,
          pulse: Math.random() * Math.PI * 2,
        });
      }
    }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      W = rect.width  || canvas.offsetWidth  || 200;
      H = rect.height || canvas.offsetHeight || height;
      canvas.width  = W;
      canvas.height = H;
      if (nodes.length === 0) initNodes();
    }

    resize();

    const ro = new ResizeObserver(() => {
      resize();
    });
    ro.observe(canvas);

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const DIST = W * 0.45;

      // update positions
      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        n.pulse += 0.03;
        if (n.x < 0) { n.x = 0; n.vx = Math.abs(n.vx); }
        if (n.x > W) { n.x = W; n.vx = -Math.abs(n.vx); }
        if (n.y < 0) { n.y = 0; n.vy = Math.abs(n.vy); }
        if (n.y > H) { n.y = H; n.vy = -Math.abs(n.vy); }
      }

      // draw edges
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < DIST) {
            const opacity = (1 - dist / DIST) * 0.45;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `rgba(0,212,255,${opacity})`;
            ctx.lineWidth = 0.8;
            ctx.stroke();
          }
        }
      }

      // draw nodes
      for (const n of nodes) {
        const glow = 0.5 + 0.5 * Math.sin(n.pulse);
        const alpha = 0.55 + 0.45 * glow;
        // outer glow ring
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0,212,255,${0.08 * glow})`;
        ctx.fill();
        // core dot
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0,212,255,${alpha})`;
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animRef.current);
      ro.disconnect();
    };
  }, [height]);

  return (
    <canvas
      ref={ref}
      style={{ width: "100%", height, display: "block" }}
    />
  );
}
