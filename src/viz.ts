import {
  DATA_XS,
  datasetLoss,
  predict,
  targetFn,
  formatNum,
  formatSigned,
  type ForwardPass,
  type Gradients,
  type Phase,
  type Weights,
} from "./net";

export interface LayoutNode {
  id: "x" | "h1" | "h2" | "y";
  label: string;
  sub: string;
  x: number;
  y: number;
  r: number;
}

export interface LayoutEdge {
  id: "w1" | "w2" | "v1" | "v2";
  from: LayoutNode["id"];
  to: LayoutNode["id"];
  weightKey: keyof Weights;
  gradKey: keyof Gradients;
}

export interface AnimState {
  /** 0→1 forward pulse progress */
  forwardT: number;
  /** 0→1 backward pulse progress */
  backwardT: number;
  /** SGD flash intensity 0→1 */
  updateFlash: number;
  /** Soft ambient breathing */
  breath: number;
  /** Intro reveal 0→1 */
  intro: number;
}

export interface DrawModel {
  weights: Weights;
  displayWeights: Weights;
  pass: ForwardPass;
  grads: Gradients | null;
  deltas: Weights | null;
  phase: Phase;
  lossHistory: number[];
  anim: AnimState;
  highlight: "none" | "forward" | "backward" | "update";
  training: boolean;
  fitted: boolean;
  trainSteps: number;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Wide: network left, fit panel right. Narrow: network on top, fit below. */
function regions(w: number, h: number): { net: Rect; fit: Rect; wide: boolean } {
  const wide = w >= 700 && w / h > 1.25;
  if (wide) {
    const fitW = Math.min(520, Math.max(300, w * 0.42));
    return {
      wide,
      net: { x: 0, y: 0, w: w - fitW - 14, h },
      fit: { x: w - fitW - 14, y: 14, w: fitW, h: h - 28 },
    };
  }
  const netH = Math.max(240, h * 0.52);
  return {
    wide,
    net: { x: 0, y: 0, w, h: netH },
    fit: { x: 10, y: netH + 2, w: w - 20, h: h - netH - 12 },
  };
}

const COLORS = {
  ink: "#e8f0f4",
  muted: "#8aa0b0",
  edge: "rgba(140, 175, 195, 0.35)",
  nodeFill: "rgba(10, 22, 32, 0.92)",
  nodeStroke: "rgba(170, 210, 225, 0.55)",
  forward: "#3ecfbf",
  forwardGlow: "rgba(62, 207, 191, 0.55)",
  backward: "#ff8a4a",
  backwardGlow: "rgba(255, 138, 74, 0.55)",
  loss: "#f5d76e",
  dead: "rgba(120, 130, 145, 0.45)",
  panel: "rgba(6, 14, 22, 0.72)",
} as const;

const EDGES: LayoutEdge[] = [
  { id: "w1", from: "x", to: "h1", weightKey: "w1", gradKey: "dLdw1" },
  { id: "w2", from: "x", to: "h2", weightKey: "w2", gradKey: "dLdw2" },
  { id: "v1", from: "h1", to: "y", weightKey: "v1", gradKey: "dLdv1" },
  { id: "v2", from: "h2", to: "y", weightKey: "v2", gradKey: "dLdv2" },
];

function layoutNodes(reg: Rect): LayoutNode[] {
  const { x: ox, y: oy, w, h } = reg;
  const r = Math.max(26, Math.min(44, Math.min(w * 0.06, h * 0.12)));
  const narrow = w < 560;
  const midY = oy + h * (narrow ? 0.44 : 0.5);
  const spread = Math.min(h * (narrow ? 0.3 : 0.29), 140);
  const x1 = ox + Math.max(14, w * 0.04) + r;
  const x3 = ox + w - r - (narrow ? 14 : Math.max(64, w * 0.13));
  const x2 = (x1 + x3) / 2;
  return [
    { id: "x", label: "x", sub: "input", x: x1, y: midY, r },
    { id: "h1", label: "h₁", sub: "ReLU", x: x2, y: midY - spread, r },
    { id: "h2", label: "h₂", sub: "ReLU", x: x2, y: midY + spread, r },
    { id: "y", label: "ŷ", sub: "linear", x: x3, y: midY, r },
  ];
}

function nodeById(nodes: LayoutNode[], id: LayoutNode["id"]): LayoutNode {
  const n = nodes.find((node) => node.id === id);
  if (!n) throw new Error(`Missing node ${id}`);
  return n;
}

function activationValue(pass: ForwardPass, id: LayoutNode["id"]): number {
  switch (id) {
    case "x":
      return pass.x;
    case "h1":
      return pass.h1;
    case "h2":
      return pass.h2;
    case "y":
      return pass.yHat;
    default: {
      const _exhaustive: never = id;
      return _exhaustive;
    }
  }
}

function isDeadReLU(pass: ForwardPass, id: LayoutNode["id"]): boolean {
  if (id === "h1") return pass.z1 <= 0;
  if (id === "h2") return pass.z2 <= 0;
  return false;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function edgePoint(
  from: LayoutNode,
  to: LayoutNode,
  t: number,
): { x: number; y: number } {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const start = from.r + 4;
  const end = len - to.r - 4;
  const d = lerp(start, end, t);
  return { x: from.x + ux * d, y: from.y + uy * d };
}

function drawEdgeBase(
  ctx: CanvasRenderingContext2D,
  from: LayoutNode,
  to: LayoutNode,
  strength: number,
): void {
  const a = edgePoint(from, to, 0);
  const b = edgePoint(from, to, 1);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = COLORS.edge;
  ctx.lineWidth = 2 + strength * 2.5;
  ctx.lineCap = "round";
  ctx.stroke();
}

function drawPulse(
  ctx: CanvasRenderingContext2D,
  from: LayoutNode,
  to: LayoutNode,
  t: number,
  color: string,
  glow: string,
  reverse: boolean,
): void {
  if (t <= 0 || t >= 1.05) return;
  const u = reverse ? 1 - easeInOut(Math.min(1, t)) : easeInOut(Math.min(1, t));
  const p = edgePoint(from, to, u);
  const trail = reverse ? -0.12 : 0.12;
  const p2 = edgePoint(from, to, Math.max(0, Math.min(1, u - trail)));

  const grad = ctx.createLinearGradient(p2.x, p2.y, p.x, p.y);
  grad.addColorStop(0, "transparent");
  grad.addColorStop(1, color);

  ctx.beginPath();
  ctx.moveTo(p2.x, p2.y);
  ctx.lineTo(p.x, p.y);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  ctx.shadowColor = glow;
  ctx.shadowBlur = 14;
  ctx.stroke();
  ctx.shadowBlur = 0;

  ctx.beginPath();
  ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.shadowColor = glow;
  ctx.shadowBlur = 18;
  ctx.fill();
  ctx.shadowBlur = 0;
}

function drawWeightLabel(
  ctx: CanvasRenderingContext2D,
  from: LayoutNode,
  to: LayoutNode,
  name: string,
  value: number,
  grad: number | null,
  flash: number,
  showGrad: boolean,
): void {
  const mid = edgePoint(from, to, 0.5);
  const top = from.id === "h1" || to.id === "h1";
  // Two-line block (value, then gradient) kept fully on the outer side of its edge.
  const narrow = Math.abs(to.x - from.x) < 150;
  const blockH = narrow ? 30 : 36;
  const bw = narrow ? 70 : 92;
  const gap = narrow ? 8 : 12;
  const y0 = top ? mid.y - gap - blockH : mid.y + gap;
  const x = mid.x;

  ctx.save();
  ctx.translate(x, y0);
  ctx.fillStyle = flash > 0.05 ? `rgba(80, 64, 20, ${0.55 + 0.35 * flash})` : "rgba(6, 14, 22, 0.78)";
  roundRect(ctx, -bw / 2, -2, bw, blockH + 4, 8);
  ctx.fill();

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `600 ${narrow ? 12 : 15}px 'IBM Plex Mono', monospace`;
  ctx.fillStyle = flash > 0.2 ? COLORS.loss : COLORS.ink;
  ctx.fillText(`${name} ${formatSigned(value)}`, 0, narrow ? 8 : 9);

  ctx.font = `600 ${narrow ? 11 : 12}px 'IBM Plex Mono', monospace`;
  ctx.fillStyle = COLORS.backward;
  ctx.globalAlpha *= showGrad && grad !== null ? 1 : 0.35;
  ctx.fillText(showGrad && grad !== null ? `∂ ${formatSigned(grad)}` : "∂ ·", 0, narrow ? 23 : 27);
  ctx.restore();
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function drawNode(
  ctx: CanvasRenderingContext2D,
  node: LayoutNode,
  value: number,
  active: boolean,
  dead: boolean,
  breath: number,
  highlight: DrawModel["highlight"],
): void {
  const pulse =
    (highlight === "forward" && active
      ? 1
      : highlight === "backward" && active
        ? 1
        : 0.35) +
    breath * 0.08;
  const r = node.r * (0.96 + pulse * 0.06);

  ctx.beginPath();
  ctx.arc(node.x, node.y, r + 10, 0, Math.PI * 2);
  const glow =
    highlight === "backward"
      ? COLORS.backwardGlow
      : highlight === "forward"
        ? COLORS.forwardGlow
        : "rgba(100, 160, 180, 0.18)";
  ctx.fillStyle = active ? glow : "rgba(80, 100, 120, 0.08)";
  ctx.fill();

  ctx.beginPath();
  ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
  ctx.fillStyle = COLORS.nodeFill;
  ctx.fill();
  ctx.lineWidth = 2.4;
  ctx.strokeStyle = dead
    ? COLORS.dead
    : highlight === "backward"
      ? COLORS.backward
      : highlight === "forward"
        ? COLORS.forward
        : COLORS.nodeStroke;
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "700 15px 'Syne', sans-serif";
  ctx.fillStyle = dead ? COLORS.muted : COLORS.ink;
  ctx.fillText(node.label, node.x, node.y - 10);

  ctx.font = "600 16px 'IBM Plex Mono', monospace";
  ctx.fillStyle = dead ? COLORS.dead : COLORS.forward;
  ctx.fillText(formatNum(value), node.x, node.y + 11);

  ctx.font = "500 10px 'Outfit', sans-serif";
  ctx.fillStyle = COLORS.muted;
  ctx.fillText(dead ? "off" : node.sub, node.x, node.y + r + 14);
}

function drawFitPanel(
  ctx: CanvasRenderingContext2D,
  r: Rect,
  model: DrawModel,
  wide: boolean,
): void {
  ctx.save();
  ctx.fillStyle = COLORS.panel;
  roundRect(ctx, r.x, r.y, r.w, r.h, 14);
  ctx.fill();
  ctx.strokeStyle = "rgba(140, 175, 195, 0.22)";
  ctx.lineWidth = 1;
  ctx.stroke();

  const pad = 14;
  const L = datasetLoss(model.displayWeights);
  // Header: title left, big loss right
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.font = "700 13px 'Outfit', sans-serif";
  ctx.fillStyle = COLORS.ink;
  ctx.fillText("Net output ŷ(x)", r.x + pad, r.y + 24);
  ctx.font = "500 11px 'Outfit', sans-serif";
  ctx.fillStyle = COLORS.loss;
  ctx.fillText("● target dots", r.x + pad, r.y + 40);
  ctx.fillStyle = COLORS.forward;
  ctx.fillText("— net curve", r.x + pad + 84, r.y + 40);

  ctx.textAlign = "right";
  ctx.font = "600 10px 'Outfit', sans-serif";
  ctx.fillStyle = COLORS.muted;
  ctx.fillText("LOSS (all 9 dots)", r.x + r.w - pad, r.y + 18);
  ctx.font = `700 ${wide ? 28 : 22}px 'IBM Plex Mono', monospace`;
  ctx.fillStyle = model.fitted ? COLORS.forward : COLORS.loss;
  ctx.fillText(formatNum(L, 3), r.x + r.w - pad, r.y + (wide ? 46 : 42));

  // Plot area
  const headH = wide ? 58 : 50;
  const lossH = Math.max(54, r.h * 0.26);
  const px = r.x + pad + 6;
  const pw = r.w - pad * 2 - 6;
  const py = r.y + headH;
  const ph = r.h - headH - lossH - 18;
  const X0 = -2.25;
  const X1 = 2.25;
  const Y0 = -0.6;
  const Y1 = 3.1;
  const sx = (x: number) => px + ((x - X0) / (X1 - X0)) * pw;
  const sy = (y: number) => py + ph - ((y - Y0) / (Y1 - Y0)) * ph;

  ctx.strokeStyle = "rgba(140, 175, 195, 0.12)";
  ctx.lineWidth = 1;
  for (let gx = -2; gx <= 2; gx += 1) {
    ctx.beginPath();
    ctx.moveTo(sx(gx), py);
    ctx.lineTo(sx(gx), py + ph);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(140, 175, 195, 0.35)";
  ctx.beginPath();
  ctx.moveTo(px, sy(0));
  ctx.lineTo(px + pw, sy(0));
  ctx.stroke();
  ctx.font = "500 10px 'IBM Plex Mono', monospace";
  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "center";
  for (const gx of [-2, -1, 0, 1, 2]) ctx.fillText(String(gx), sx(gx), py + ph + 12);

  ctx.save();
  ctx.beginPath();
  ctx.rect(px - 4, py - 4, pw + 8, ph + 8);
  ctx.clip();

  // Target (faint guide) + dots
  ctx.setLineDash([4, 5]);
  ctx.strokeStyle = "rgba(245, 215, 110, 0.35)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(sx(X0), sy(targetFn(X0)));
  ctx.lineTo(sx(0), sy(0));
  ctx.lineTo(sx(X1), sy(targetFn(X1)));
  ctx.stroke();
  ctx.setLineDash([]);

  // Net curve
  const wts = model.displayWeights;
  ctx.beginPath();
  for (let i = 0; i <= 90; i += 1) {
    const x = X0 + ((X1 - X0) * i) / 90;
    const y = predict(wts, x);
    if (i === 0) ctx.moveTo(sx(x), sy(y));
    else ctx.lineTo(sx(x), sy(y));
  }
  ctx.strokeStyle = COLORS.forward;
  ctx.lineWidth = 4;
  ctx.lineJoin = "round";
  ctx.shadowColor = COLORS.forwardGlow;
  ctx.shadowBlur = 16;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Residual of the current sample (coral), only once a pass has run
  const cur = model.pass.x;
  const isSample = DATA_XS.some((d) => Math.abs(d - cur) < 1e-6);
  if (model.phase !== "idle") {
    const yh = predict(wts, cur);
    const tt = model.pass.t;
    ctx.strokeStyle = COLORS.backward;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(sx(cur), sy(yh));
    ctx.lineTo(sx(cur), sy(tt));
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(sx(cur), sy(yh), 6, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.forward;
    ctx.fill();
    if (!isSample) {
      ctx.beginPath();
      ctx.arc(sx(cur), sy(tt), 5, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.backward;
      ctx.fill();
    }
  }

  for (const d of DATA_XS) {
    const active = Math.abs(d - cur) < 1e-6 && model.phase !== "idle";
    ctx.beginPath();
    ctx.arc(sx(d), sy(targetFn(d)), active ? 7.5 : 5.5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.loss;
    ctx.shadowColor = "rgba(245, 215, 110, 0.6)";
    ctx.shadowBlur = active ? 14 : 6;
    ctx.fill();
    ctx.shadowBlur = 0;
    if (active) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#fff6c2";
      ctx.stroke();
    }
  }
  ctx.restore();

  // Loss-over-steps chart (log scale so the dive stays visible)
  const lx = px;
  const lw = pw;
  const ly = py + ph + 22;
  const lh = r.y + r.h - ly - 10;
  ctx.fillStyle = "rgba(245, 215, 110, 0.05)";
  roundRect(ctx, lx, ly, lw, lh, 8);
  ctx.fill();
  ctx.textAlign = "left";
  ctx.font = "600 10px 'Outfit', sans-serif";
  ctx.fillStyle = COLORS.muted;
  ctx.fillText("loss per SGD step (log)", lx + 8, ly + 14);
  const hist = model.lossHistory;
  if (hist.length >= 2) {
    const lg = (v: number) => Math.log10(Math.max(v, 1e-4));
    const hi = Math.max(...hist.map(lg), -1);
    const lo = Math.min(hi - 1, -3);
    const n = Math.max(hist.length - 1, 24);
    ctx.beginPath();
    hist.forEach((v, i) => {
      const xx = lx + 8 + (i / n) * (lw - 16);
      const yy = ly + 20 + (1 - (lg(v) - lo) / (hi - lo)) * (lh - 26);
      if (i === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.strokeStyle = COLORS.loss;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    const i = hist.length - 1;
    const xx = lx + 8 + (i / n) * (lw - 16);
    const yy = ly + 20 + (1 - (lg(hist[i]!) - lo) / (hi - lo)) * (lh - 26);
    ctx.beginPath();
    ctx.arc(xx, yy, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#fff6c2";
    ctx.fill();
  }
  ctx.restore();
}

function drawTargetMark(
  ctx: CanvasRenderingContext2D,
  yNode: LayoutNode,
  t: number,
  yHat: number,
): void {
  if (yNode.x + yNode.r + 90 > ctx.canvas.width / Math.max(1, ctx.getTransform().a)) {
    ctx.font = "600 12px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillStyle = COLORS.backward;
    ctx.fillText(`t ${formatNum(t)}`, yNode.x, yNode.y + yNode.r + 30);
    return;
  }
  const x = yNode.x + yNode.r + 28;
  const y = yNode.y;
  ctx.font = "600 12px 'Outfit', sans-serif";
  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "left";
  ctx.fillText("target t", x, y - 18);
  ctx.font = "700 18px 'IBM Plex Mono', monospace";
  ctx.fillStyle = COLORS.backward;
  ctx.fillText(formatNum(t), x, y + 2);

  const err = yHat - t;
  ctx.font = "500 11px 'Outfit', sans-serif";
  ctx.fillStyle = COLORS.muted;
  ctx.fillText(`err ${formatSigned(err)}`, x, y + 22);
}

function phaseAlpha(anim: AnimState, phase: Phase, kind: "forward" | "backward"): number {
  if (kind === "forward") {
    if (phase === "forward" || phase === "backward" || phase === "updated") {
      return Math.min(1, anim.forwardT);
    }
    return 0.15 + anim.breath * 0.1;
  }
  if (phase === "backward" || phase === "updated") {
    return Math.min(1, anim.backwardT);
  }
  return 0;
}

export function paintStage(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  dpr: number,
  model: DrawModel,
): void {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const intro = model.anim.intro;
  ctx.save();
  ctx.globalAlpha = intro;

  // Soft grid wash
  ctx.strokeStyle = "rgba(120, 170, 190, 0.05)";
  ctx.lineWidth = 1;
  const step = 36;
  for (let gx = 0; gx < width; gx += step) {
    ctx.beginPath();
    ctx.moveTo(gx, 0);
    ctx.lineTo(gx, height);
    ctx.stroke();
  }
  for (let gy = 0; gy < height; gy += step) {
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(width, gy);
    ctx.stroke();
  }

  const reg = regions(width, height);
  const nodes = layoutNodes(reg.net);
  const map = {
    x: nodeById(nodes, "x"),
    h1: nodeById(nodes, "h1"),
    h2: nodeById(nodes, "h2"),
    y: nodeById(nodes, "y"),
  };

  const showGrad =
    model.phase === "backward" ||
    model.phase === "updated" ||
    model.highlight === "backward";
  const fwdA = phaseAlpha(model.anim, model.phase, "forward");

  for (const edge of EDGES) {
    const from = map[edge.from];
    const to = map[edge.to];
    const wVal = model.displayWeights[edge.weightKey];
    const strength = Math.min(1.4, Math.abs(wVal));
    drawEdgeBase(ctx, from, to, strength);

    if (model.highlight === "forward" || model.phase === "forward") {
      const delay =
        edge.id === "w1" || edge.id === "w2"
          ? 0
          : 0.35;
      const local = Math.max(0, Math.min(1, (model.anim.forwardT - delay) / 0.55));
      drawPulse(ctx, from, to, local, COLORS.forward, COLORS.forwardGlow, false);
    }

    if (model.highlight === "backward" || model.phase === "backward") {
      const delay =
        edge.id === "v1" || edge.id === "v2"
          ? 0
          : 0.35;
      const local = Math.max(0, Math.min(1, (model.anim.backwardT - delay) / 0.55));
      drawPulse(ctx, from, to, local, COLORS.backward, COLORS.backwardGlow, true);
    }

    const grad =
      model.grads && showGrad ? model.grads[edge.gradKey] : null;
    const flash =
      model.deltas && model.anim.updateFlash > 0
        ? model.anim.updateFlash
        : 0;
    drawWeightLabel(
      ctx,
      from,
      to,
      edge.id,
      wVal,
      typeof grad === "number" ? grad : null,
      flash,
      showGrad && fwdA > 0.5,
    );
  }

  for (const node of nodes) {
    const value = activationValue(model.pass, node.id);
    const dead = isDeadReLU(model.pass, node.id);
    const active =
      model.highlight === "forward"
        ? model.anim.forwardT > (node.id === "x" ? 0 : node.id === "y" ? 0.55 : 0.25)
        : model.highlight === "backward"
          ? model.anim.backwardT > (node.id === "y" ? 0 : node.id === "x" ? 0.55 : 0.25)
          : model.phase !== "idle" || true;
    drawNode(
      ctx,
      node,
      value,
      active,
      dead,
      model.anim.breath,
      model.highlight === "none" ? "none" : model.highlight,
    );
  }

  drawTargetMark(ctx, map.y, model.pass.t, model.pass.yHat);
  drawFitPanel(ctx, reg.fit, model, reg.wide);

  // Phase caption strip
  ctx.font = "600 12px 'Outfit', sans-serif";
  ctx.textAlign = "left";
  ctx.fillStyle = COLORS.muted;
  const caption = model.training
    ? `Training · sample x = ${formatSigned(model.pass.x)} · step ${model.trainSteps}`
    : model.fitted
      ? `Fitted in ${model.trainSteps} steps · Randomize to scramble`
      : model.phase === "forward"
      ? "Forward · activations lit · loss from ŷ vs t"
      : model.phase === "backward"
        ? "Backward · ∂L/∂w on every edge · same weights"
        : model.phase === "updated"
          ? "SGD applied · weights jumped · forward re-run"
          : "Idle · tweak x / t, then Step Forward";
  const capY = reg.net.y + reg.net.h - (reg.wide ? 14 : 4);
  ctx.fillText(caption, reg.net.x + 16, capY);

  // Tiny legend (top-left of the network area)
  ctx.textAlign = "left";
  if (!reg.wide) ctx.globalAlpha = 0;
  ctx.fillStyle = COLORS.forward;
  ctx.fillText("→ forward signal", reg.net.x + 16, reg.net.y + 22);
  ctx.fillStyle = COLORS.backward;
  ctx.fillText("← gradient", reg.net.x + 16, reg.net.y + 38);

  ctx.restore();
}

export { COLORS, EDGES };
