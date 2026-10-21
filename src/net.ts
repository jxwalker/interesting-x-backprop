/** Tiny net: 1 input → 2 ReLU hidden → 1 linear output. Four weights only. */

export interface Weights {
  w1: number;
  w2: number;
  v1: number;
  v2: number;
}

export interface ForwardPass {
  x: number;
  t: number;
  z1: number;
  z2: number;
  h1: number;
  h2: number;
  yHat: number;
  loss: number;
}

export interface Gradients {
  dLdy: number;
  dLdv1: number;
  dLdv2: number;
  dLdh1: number;
  dLdh2: number;
  dLdz1: number;
  dLdz2: number;
  dLdw1: number;
  dLdw2: number;
}

export type Phase = "idle" | "forward" | "backward" | "updated";

export const DEFAULT_WEIGHTS: Weights = {
  w1: 0.8,
  w2: -0.55,
  v1: 0.65,
  v2: 0.9,
};

export const DEFAULT_X = 1.4;
export const DEFAULT_T = 1.95;
export const DEFAULT_LR = 0.08;
export const LR_MIN = 0.02;
export const LR_MAX = 0.3;
export const X_MIN = -2;
export const X_MAX = 2;
export const T_MIN = -1.5;
export const T_MAX = 3;
export const MAX_LOSS_HISTORY = 90;

export function relu(z: number): number {
  return z > 0 ? z : 0;
}

export function reluPrime(z: number): number {
  return z > 0 ? 1 : 0;
}

export function forward(weights: Weights, x: number, t: number): ForwardPass {
  const z1 = weights.w1 * x;
  const z2 = weights.w2 * x;
  const h1 = relu(z1);
  const h2 = relu(z2);
  const yHat = weights.v1 * h1 + weights.v2 * h2;
  const err = yHat - t;
  const loss = 0.5 * err * err;
  return { x, t, z1, z2, h1, h2, yHat, loss };
}

export function backward(weights: Weights, pass: ForwardPass): Gradients {
  const dLdy = pass.yHat - pass.t;
  const dLdv1 = dLdy * pass.h1;
  const dLdv2 = dLdy * pass.h2;
  const dLdh1 = dLdy * weights.v1;
  const dLdh2 = dLdy * weights.v2;
  const dLdz1 = dLdh1 * reluPrime(pass.z1);
  const dLdz2 = dLdh2 * reluPrime(pass.z2);
  const dLdw1 = dLdz1 * pass.x;
  const dLdw2 = dLdz2 * pass.x;
  return {
    dLdy,
    dLdv1,
    dLdv2,
    dLdh1,
    dLdh2,
    dLdz1,
    dLdz2,
    dLdw1,
    dLdw2,
  };
}

/** SGD at the original parameter values (all four grads applied together). */
export function applySgd(weights: Weights, grads: Gradients, lr: number): Weights {
  return {
    w1: weights.w1 - lr * grads.dLdw1,
    w2: weights.w2 - lr * grads.dLdw2,
    v1: weights.v1 - lr * grads.dLdv1,
    v2: weights.v2 - lr * grads.dLdv2,
  };
}

export function weightDeltas(
  before: Weights,
  after: Weights,
): Weights {
  return {
    w1: after.w1 - before.w1,
    w2: after.w2 - before.w2,
    v1: after.v1 - before.v1,
    v2: after.v2 - before.v2,
  };
}

/** Deterministic-ish randomize in a playful range that usually trains. */
export function randomWeights(seed = Math.random()): Weights {
  const r = (offset: number) => {
    const u = Math.abs(Math.sin(seed * 9999 + offset * 17.13));
    return (u * 2 - 1) * 1.15;
  };
  // Keep one ReLU per side alive so every scramble is trainable, but start far
  // enough from the target that the fit visibly bends into place.
  const u = (offset: number, lo: number, hi: number) =>
    lo + (hi - lo) * ((r(offset) / 1.15 + 1) / 2);
  return {
    w1: u(1, 0.4, 1.2),
    w2: u(2, -1.2, -0.4),
    v1: u(3, -0.5, 0.6),
    v2: u(4, 0.5, 1.3),
  };
}

export function formatNum(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  let body: string;
  if (abs >= 100) body = n.toFixed(0);
  else if (abs >= 10) body = n.toFixed(1);
  else body = n.toFixed(digits);
  if (body === "-0.00" || body === "-0.0") body = digits === 1 ? "0.0" : "0.00";
  return body;
}

export function formatSigned(n: number, digits = 2): string {
  const body = formatNum(n, digits);
  if (body === "—" || body.startsWith("-") || n === 0 || body === "0.00" || body === "0.0") {
    return body;
  }
  return `+${body}`;
}

/** Training set: the net must bend its two ReLU arms onto this "hockey stick". */
export const TARGET_RIGHT = 1.4;
export const TARGET_LEFT = 0.15;
export const DATA_XS: readonly number[] = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];

export function targetFn(x: number): number {
  return TARGET_RIGHT * relu(x) + TARGET_LEFT * relu(-x);
}

export function predict(weights: Weights, x: number): number {
  return weights.v1 * relu(weights.w1 * x) + weights.v2 * relu(weights.w2 * x);
}

/** Mean ½(ŷ − t)² over the whole training set. */
export function datasetLoss(weights: Weights): number {
  let sum = 0;
  for (const x of DATA_XS) {
    const e = predict(weights, x) - targetFn(x);
    sum += 0.5 * e * e;
  }
  return sum / DATA_XS.length;
}
