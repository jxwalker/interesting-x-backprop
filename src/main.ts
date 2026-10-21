import "./styles.css";
import {
  DEFAULT_LR,
  DEFAULT_T,
  DEFAULT_WEIGHTS,
  DEFAULT_X,
  LR_MAX,
  LR_MIN,
  MAX_LOSS_HISTORY,
  T_MAX,
  T_MIN,
  X_MAX,
  X_MIN,
  DATA_XS,
  applySgd,
  backward,
  datasetLoss,
  targetFn,
  formatNum,
  forward,
  randomWeights,
  weightDeltas,
  type ForwardPass,
  type Gradients,
  type Phase,
  type Weights,
} from "./net";
import { paintStage, type AnimState, type DrawModel } from "./viz";

interface AppState {
  weights: Weights;
  displayWeights: Weights;
  x: number;
  t: number;
  lr: number;
  pass: ForwardPass;
  grads: Gradients | null;
  deltas: Weights | null;
  phase: Phase;
  lossHistory: number[];
  anim: AnimState;
  highlight: DrawModel["highlight"];
  training: boolean;
  fitted: boolean;
  trainSteps: number;
}

function mustEl<T extends HTMLElement>(
  root: HTMLElement,
  selector: string,
  ctor: new () => T,
): T {
  const el = root.querySelector(selector);
  if (!(el instanceof ctor)) {
    throw new Error(`Missing or wrong element: ${selector}`);
  }
  return el;
}

function isRecordMode(): boolean {
  return new URLSearchParams(window.location.search).has("record");
}

function setText(el: HTMLElement, next: string): void {
  if (el.textContent !== next) el.textContent = next;
}

function phaseLabel(phase: Phase): string {
  switch (phase) {
    case "idle":
      return "Idle";
    case "forward":
      return "Forward";
    case "backward":
      return "Backward";
    case "updated":
      return "Updated";
    default: {
      const _exhaustive: never = phase;
      return _exhaustive;
    }
  }
}

function cloneWeights(w: Weights): Weights {
  return { w1: w.w1, w2: w.w2, v1: w.v1, v2: w.v2 };
}

function lerpWeights(a: Weights, b: Weights, t: number): Weights {
  return {
    w1: a.w1 + (b.w1 - a.w1) * t,
    w2: a.w2 + (b.w2 - a.w2) * t,
    v1: a.v1 + (b.v1 - a.v1) * t,
    v2: a.v2 + (b.v2 - a.v2) * t,
  };
}

function mount(): void {
  const appRoot = document.getElementById("app");
  if (!(appRoot instanceof HTMLElement)) {
    throw new Error("Missing #app");
  }
  const root = appRoot;
  const recordMode = isRecordMode();

  const initialPass = forward(DEFAULT_WEIGHTS, DEFAULT_X, DEFAULT_T);

  const state: AppState = {
    weights: cloneWeights(DEFAULT_WEIGHTS),
    displayWeights: cloneWeights(DEFAULT_WEIGHTS),
    x: DEFAULT_X,
    t: DEFAULT_T,
    lr: DEFAULT_LR,
    pass: initialPass,
    grads: null,
    deltas: null,
    phase: "idle",
    lossHistory: [datasetLoss(DEFAULT_WEIGHTS)],
    anim: {
      forwardT: 0,
      backwardT: 0,
      updateFlash: 0,
      breath: 0,
      intro: recordMode ? 1 : 0,
    },
    highlight: "none",
    training: false,
    fitted: false,
    trainSteps: 0,
  };

  root.innerHTML = `
    <div class="grain" aria-hidden="true"></div>
    <div class="wash" aria-hidden="true"></div>

    <header class="hero">
      <div class="brand-block">
        <p class="brand">Interesting X</p>
        <p class="title">Backprop by hand · 1→2 ReLU→1</p>
      </div>
      <div class="loss-chip" id="loss-chip">
        <span class="label">Loss</span>
        <span class="value" id="loss-value">${formatNum(initialPass.loss, 3)}</span>
      </div>
    </header>

    <p class="hint" id="hint">
      Four weights, nine dots. <strong>Train</strong> and watch backprop bend the net&rsquo;s curve onto the target,
      or step one pass at a time.
    </p>

    <div class="stage">
      <canvas id="canvas" aria-label="Tiny neural network graph with live weights and activations"></canvas>
    </div>

    <section class="controls" aria-label="Backprop controls">
      <div class="panel">
        <p class="panel-title">Signals</p>
        <div class="sliders">
          <div class="slider-row">
            <label for="param-x">Input x</label>
            <input id="param-x" type="range" min="${X_MIN}" max="${X_MAX}" step="0.05" value="${DEFAULT_X}" />
            <output id="out-x" for="param-x">${formatNum(DEFAULT_X)}</output>
          </div>
          <div class="slider-row">
            <label for="param-t">Target t</label>
            <input id="param-t" type="range" min="${T_MIN}" max="${T_MAX}" step="0.05" value="${DEFAULT_T}" />
            <output id="out-t" for="param-t">${formatNum(DEFAULT_T)}</output>
          </div>
          <div class="slider-row">
            <label for="param-lr">Learn rate</label>
            <input id="param-lr" type="range" min="${LR_MIN}" max="${LR_MAX}" step="0.01" value="${DEFAULT_LR}" />
            <output id="out-lr" for="param-lr">${formatNum(DEFAULT_LR)}</output>
          </div>
        </div>
      </div>

      <div class="panel">
        <p class="panel-title">Pass</p>
        <div class="steps">
          <button type="button" class="btn btn-train" id="btn-train">&#9654; Train</button>
          <button type="button" class="btn btn-forward" id="btn-forward">Step Forward</button>
          <button type="button" class="btn btn-backward" id="btn-backward" disabled>Step Backward</button>
          <button type="button" class="btn btn-sgd" id="btn-sgd" disabled>Apply SGD</button>
        </div>
      </div>

      <div class="util">
        <span class="phase-pill" id="phase-pill" data-phase="idle">
          <span class="phase-dot" aria-hidden="true"></span>
          <span id="phase-text">Idle</span>
        </span>
        <button type="button" class="btn btn-ghost" id="btn-random">Randomize weights</button>
        <button type="button" class="btn btn-ghost" id="btn-reset">Reset</button>
      </div>
    </section>

    <p class="foot">All four ∂L/∂w use the same weights — then one SGD jump. Dead ReLUs show “off”.</p>
  `;

  const canvas = mustEl(root, "#canvas", HTMLCanvasElement);
  const canvasCtx = canvas.getContext("2d");
  if (!canvasCtx) throw new Error("Canvas 2D unavailable");
  const ctx: CanvasRenderingContext2D = canvasCtx;

  const lossChip = mustEl(root, "#loss-chip", HTMLDivElement);
  const lossValue = mustEl(root, "#loss-value", HTMLSpanElement);
  const phasePill = mustEl(root, "#phase-pill", HTMLSpanElement);
  const phaseText = mustEl(root, "#phase-text", HTMLSpanElement);
  const outX = mustEl(root, "#out-x", HTMLOutputElement);
  const outT = mustEl(root, "#out-t", HTMLOutputElement);
  const outLr = mustEl(root, "#out-lr", HTMLOutputElement);
  const paramX = mustEl(root, "#param-x", HTMLInputElement);
  const paramT = mustEl(root, "#param-t", HTMLInputElement);
  const paramLr = mustEl(root, "#param-lr", HTMLInputElement);
  const btnForward = mustEl(root, "#btn-forward", HTMLButtonElement);
  const btnBackward = mustEl(root, "#btn-backward", HTMLButtonElement);
  const btnSgd = mustEl(root, "#btn-sgd", HTMLButtonElement);
  const btnTrain = mustEl(root, "#btn-train", HTMLButtonElement);
  const btnRandom = mustEl(root, "#btn-random", HTMLButtonElement);
  const btnReset = mustEl(root, "#btn-reset", HTMLButtonElement);

  let animToken = 0;
  let trainToken = 0;
  let trainOrder: number[] = [];
  let weightTween: { from: Weights; to: Weights; t: number } | null = null;

  function syncChrome(): void {
    setText(lossValue, formatNum(datasetLoss(state.weights), 3));
    setText(btnTrain, state.training ? "❚❚ Pause" : state.fitted ? "✓ Fitted · Train" : "▶ Train");
    btnTrain.classList.toggle("active", state.training);
    setText(phaseText, phaseLabel(state.phase));
    phasePill.dataset.phase = state.phase;
    setText(outX, formatNum(state.x));
    setText(outT, formatNum(state.t));
    setText(outLr, formatNum(state.lr));

    btnForward.classList.toggle("active", state.phase === "forward");
    btnBackward.classList.toggle("active", state.phase === "backward");
    btnSgd.classList.toggle("active", state.phase === "updated");

    const canBack = state.phase === "forward" || state.phase === "backward" || state.phase === "updated";
    const canSgd = state.phase === "backward" || state.phase === "updated";
    btnBackward.disabled = !canBack;
    btnSgd.disabled = !canSgd;
  }

  function recomputeForward(pushHistory: boolean): void {
    state.pass = forward(state.weights, state.x, state.t);
    if (pushHistory) {
      state.lossHistory = [...state.lossHistory, datasetLoss(state.weights)].slice(-MAX_LOSS_HISTORY);
    } else if (state.lossHistory.length === 0) {
      state.lossHistory = [datasetLoss(state.weights)];
    }
    if (state.phase === "backward" || state.phase === "updated") {
      state.grads = backward(state.weights, state.pass);
    }
    syncChrome();
  }

  function runAnim(
    kind: "forward" | "backward" | "update",
    durationMs: number,
  ): Promise<void> {
    const token = ++animToken;
    const start = performance.now();
    state.highlight = kind === "update" ? "update" : kind;

    return new Promise((resolve) => {
      const tick = (now: number): void => {
        if (token !== animToken) {
          resolve();
          return;
        }
        const u = Math.min(1, (now - start) / durationMs);
        if (kind === "forward") state.anim.forwardT = u;
        if (kind === "backward") state.anim.backwardT = u;
        if (kind === "update") state.anim.updateFlash = 1 - u;
        if (u < 1) {
          requestAnimationFrame(tick);
        } else {
          if (kind === "forward") state.anim.forwardT = 1;
          if (kind === "backward") state.anim.backwardT = 1;
          if (kind === "update") state.anim.updateFlash = 0;
          if (kind !== "update") state.highlight = kind;
          else state.highlight = "none";
          resolve();
        }
      };
      requestAnimationFrame(tick);
    });
  }

  async function stepForward(ms?: number): Promise<void> {
    state.grads = null;
    state.deltas = null;
    state.phase = "forward";
    state.anim.backwardT = 0;
    state.anim.updateFlash = 0;
    recomputeForward(false);
    syncChrome();
    await runAnim("forward", ms ?? (recordMode ? 900 : 1100));
    syncChrome();
  }

  async function stepBackward(ms?: number): Promise<void> {
    if (state.phase === "idle") {
      await stepForward();
    }
    state.pass = forward(state.weights, state.x, state.t);
    state.grads = backward(state.weights, state.pass);
    state.deltas = null;
    state.phase = "backward";
    state.anim.forwardT = 1;
    syncChrome();
    await runAnim("backward", ms ?? (recordMode ? 1000 : 1200));
    syncChrome();
  }

  async function applyStep(ms?: number): Promise<void> {
    if (!state.grads) {
      await stepBackward();
    }
    const grads = state.grads ?? backward(state.weights, state.pass);
    const before = cloneWeights(state.weights);
    // Grads frozen at original params, then one joint update.
    const after = applySgd(before, grads, state.lr);
    state.deltas = weightDeltas(before, after);
    state.grads = grads;
    weightTween = { from: before, to: after, t: 0 };
    state.weights = after;
    state.phase = "updated";
    state.anim.updateFlash = 1;
    lossChip.classList.remove("flash");
    void lossChip.offsetWidth;
    lossChip.classList.add("flash");

    await runAnim("update", ms ?? (recordMode ? 700 : 850));

    // Re-run forward so loss visibly moves.
    state.displayWeights = cloneWeights(after);
    weightTween = null;
    recomputeForward(true);
    state.grads = backward(state.weights, state.pass);
    state.anim.forwardT = 1;
    state.anim.backwardT = 1;
    syncChrome();
  }

  function stopTraining(): void {
    trainToken += 1;
    state.training = false;
  }

  function setSample(x: number, t: number): void {
    state.x = x;
    state.t = Math.max(T_MIN, Math.min(T_MAX, t));
    paramX.value = String(state.x);
    paramT.value = String(state.t);
  }

  async function trainLoop(): Promise<void> {
    const token = ++trainToken;
    state.training = true;
    state.fitted = false;
    syncChrome();
    let n = 0;
    while (token === trainToken) {
      if (trainOrder.length === 0) {
        trainOrder = [...DATA_XS];
        for (let i = trainOrder.length - 1; i > 0; i -= 1) {
          const j = Math.floor(Math.random() * (i + 1));
          [trainOrder[i], trainOrder[j]] = [trainOrder[j]!, trainOrder[i]!];
        }
      }
      const x = trainOrder.pop()!;
      setSample(x, targetFn(x));
      // Slow, legible first passes, then speed up so the loss visibly dives.
      const dur = Math.max(280, 900 * Math.pow(0.92, n));
      await stepForward(dur * 0.38);
      if (token !== trainToken) break;
      await stepBackward(dur * 0.38);
      if (token !== trainToken) break;
      await applyStep(dur * 0.24);
      if (token !== trainToken) break;
      n += 1;
      state.trainSteps += 1;
      const L = datasetLoss(state.weights);
      if (!Number.isFinite(L) || L > 50) {
        // Diverged (learn rate too high): start over from fresh weights.
        state.weights = randomWeights(performance.now() * 0.001);
        state.displayWeights = cloneWeights(state.weights);
        state.lossHistory = [datasetLoss(state.weights)];
      } else if (L < 0.003) {
        state.fitted = true;
        break;
      } else if (n >= 160) {
        break;
      }
    }
    if (token === trainToken) state.training = false;
    syncChrome();
  }

  btnTrain.addEventListener("click", () => {
    if (state.training) {
      stopTraining();
      syncChrome();
    } else {
      void trainLoop();
    }
  });

  function resetAll(): void {
    stopTraining();
    state.fitted = false;
    state.trainSteps = 0;
    animToken += 1;
    state.weights = cloneWeights(DEFAULT_WEIGHTS);
    state.displayWeights = cloneWeights(DEFAULT_WEIGHTS);
    state.x = DEFAULT_X;
    state.t = DEFAULT_T;
    state.lr = DEFAULT_LR;
    paramX.value = String(DEFAULT_X);
    paramT.value = String(DEFAULT_T);
    paramLr.value = String(DEFAULT_LR);
    state.grads = null;
    state.deltas = null;
    state.phase = "idle";
    state.highlight = "none";
    state.anim.forwardT = 0;
    state.anim.backwardT = 0;
    state.anim.updateFlash = 0;
    weightTween = null;
    state.pass = forward(state.weights, state.x, state.t);
    state.lossHistory = [datasetLoss(state.weights)];
    syncChrome();
  }

  function randomize(): void {
    stopTraining();
    state.fitted = false;
    state.trainSteps = 0;
    animToken += 1;
    state.weights = randomWeights(performance.now() * 0.001);
    state.displayWeights = cloneWeights(state.weights);
    state.grads = null;
    state.deltas = null;
    state.phase = "idle";
    state.highlight = "none";
    state.anim.forwardT = 0;
    state.anim.backwardT = 0;
    state.anim.updateFlash = 0;
    weightTween = null;
    state.pass = forward(state.weights, state.x, state.t);
    state.lossHistory = [datasetLoss(state.weights)];
    syncChrome();
    void trainLoop();
  }

  paramX.addEventListener("input", () => {
    stopTraining();
    state.x = Number(paramX.value);
    if (state.phase === "idle") {
      recomputeForward(false);
    } else {
      state.phase = "forward";
      state.grads = null;
      state.deltas = null;
      state.anim.forwardT = 1;
      state.anim.backwardT = 0;
      recomputeForward(false);
    }
    syncChrome();
  });

  paramT.addEventListener("input", () => {
    stopTraining();
    state.t = Number(paramT.value);
    if (state.phase === "idle") {
      recomputeForward(false);
    } else {
      state.phase = "forward";
      state.grads = null;
      state.deltas = null;
      state.anim.forwardT = 1;
      state.anim.backwardT = 0;
      recomputeForward(false);
    }
    syncChrome();
  });

  paramLr.addEventListener("input", () => {
    state.lr = Number(paramLr.value);
    syncChrome();
  });

  btnForward.addEventListener("click", () => {
    stopTraining();
    void stepForward();
  });
  btnBackward.addEventListener("click", () => {
    stopTraining();
    void stepBackward();
  });
  btnSgd.addEventListener("click", () => {
    stopTraining();
    void applyStep();
  });
  btnReset.addEventListener("click", resetAll);
  btnRandom.addEventListener("click", randomize);

  function resize(): void {
    const stage = canvas.parentElement;
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(320, Math.floor(rect.width));
    const h = Math.max(280, Math.floor(rect.height));
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }

  let lastTs = performance.now();
  function frame(now: number): void {
    const dt = Math.min(0.05, (now - lastTs) / 1000);
    lastTs = now;

    if (!recordMode && state.anim.intro < 1) {
      state.anim.intro = Math.min(1, state.anim.intro + dt * 1.4);
    }
    state.anim.breath = 0.5 + 0.5 * Math.sin(now * 0.0018);

    if (weightTween) {
      weightTween.t = Math.min(1, weightTween.t + dt * 2.4);
      state.displayWeights = lerpWeights(
        weightTween.from,
        weightTween.to,
        weightTween.t,
      );
      if (weightTween.t >= 1) {
        state.displayWeights = cloneWeights(weightTween.to);
        weightTween = null;
      }
    } else {
      state.displayWeights = cloneWeights(state.weights);
    }

    const stage = canvas.parentElement;
    const rect = stage?.getBoundingClientRect();
    const w = rect ? Math.max(320, Math.floor(rect.width)) : 640;
    const h = rect ? Math.max(280, Math.floor(rect.height)) : 400;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    // Keep the backing store in sync with late layout changes (fonts, Safari).
    if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
      resize();
    }

    paintStage(ctx, w, h, dpr, {
      weights: state.weights,
      displayWeights: state.displayWeights,
      pass: state.pass,
      grads: state.grads,
      deltas: state.deltas,
      phase: state.phase,
      lossHistory: state.lossHistory,
      anim: state.anim,
      highlight: state.highlight,
      training: state.training,
      fitted: state.fitted,
      trainSteps: state.trainSteps,
    });

    requestAnimationFrame(frame);
  }

  resize();
  syncChrome();
  window.addEventListener("resize", resize);
  requestAnimationFrame(frame);

  // Auto-demo pulse on load so first impression isn't static.
  if (!recordMode) {
    window.setTimeout(() => {
      if (state.phase === "idle" && !state.training) void trainLoop();
    }, 450);
  } else {
    // Kick a gentle forward so the OG / first frame isn't empty.
    state.anim.forwardT = 0.35;
    state.phase = "forward";
    syncChrome();
  }
}

mount();
