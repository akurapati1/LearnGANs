"""Train a vanilla (non-saturating) GAN on a ring of Gaussians and measure mode coverage."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .data import ring_modes, sample_real
from .nn import MLP


@dataclass
class Config:
    steps: int = 4000
    batch: int = 256
    z_dim: int = 8
    hidden: int = 128
    lr_g: float = 1e-3
    lr_d: float = 1e-3
    d_steps: int = 1
    n_modes: int = 8
    std: float = 0.1
    seed: int = 0


def evaluate(samples: np.ndarray, n_modes: int = 8, std: float = 0.1) -> dict:
    """Mode coverage = modes with >=1% of samples within 3 sigma; quality = share of samples near any mode."""
    d = np.linalg.norm(samples[:, None, :] - ring_modes(n_modes)[None], axis=2)
    near = d < 3 * std * np.sqrt(2)
    hq = float(near.any(1).mean())
    covered = int((near.sum(0) >= 0.01 * len(samples)).sum())
    return {"modes_covered": covered, "modes_total": n_modes, "high_quality": round(hq, 3)}


def train(cfg: Config, log=lambda *_: None) -> tuple[MLP, MLP, list[dict]]:
    rng = np.random.default_rng(cfg.seed)
    G = MLP([cfg.z_dim, cfg.hidden, cfg.hidden, 2], rng)
    D = MLP([2, cfg.hidden, cfg.hidden, 1], rng, out="sigmoid")
    history = []
    for step in range(1, cfg.steps + 1):
        for _ in range(cfg.d_steps):
            real = sample_real(cfg.batch, rng, cfg.n_modes, cfg.std)
            fake = G.forward(rng.normal(size=(cfg.batch, cfg.z_dim)))
            x = np.vstack([real, fake])
            y = np.vstack([np.full((cfg.batch, 1), 0.9), np.zeros((cfg.batch, 1))])  # one-sided label smoothing
            p = D.forward(x)
            grads, _ = D.backward((p - y) / len(x))  # BCE + sigmoid => p - y
            D.step(grads, cfg.lr_d)
        z = rng.normal(size=(cfg.batch, cfg.z_dim))
        fake = G.forward(z)
        p = D.forward(fake)
        # non-saturating generator loss -log D(G(z)); dL/dlogit = p - 1
        _, dx = D.backward((p - 1.0) / cfg.batch)
        gg, _ = G.backward(dx)
        G.step(gg, cfg.lr_g)
        if step % 500 == 0 or step == cfg.steps:
            m = evaluate(G.forward(rng.normal(size=(2000, cfg.z_dim))), cfg.n_modes, cfg.std)
            history.append({"step": step, **m})
            log(f"step {step:5d}  modes {m['modes_covered']}/{m['modes_total']}  quality {m['high_quality']:.0%}")
    return G, D, history


def to_svg(real: np.ndarray, fake: np.ndarray, size: int = 480) -> str:
    def pt(a, color, r):
        s = (a + 3.2) / 6.4 * size
        return "".join(f'<circle cx="{x:.1f}" cy="{size - y:.1f}" r="{r}" fill="{color}" fill-opacity="0.45"/>' for x, y in s)

    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<rect width="100%" height="100%" fill="#0f172a"/>{pt(real, "#38bdf8", 2)}{pt(fake, "#f97316", 2)}'
            f'<text x="12" y="22" fill="#38bdf8" font-family="monospace" font-size="13">real</text>'
            f'<text x="60" y="22" fill="#f97316" font-family="monospace" font-size="13">generated</text></svg>')


def main() -> None:
    ap = argparse.ArgumentParser(description="NumPy GAN on a ring of 8 Gaussians")
    ap.add_argument("--steps", type=int, default=Config.steps)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--svg", default="samples.svg")
    a = ap.parse_args()
    cfg = Config(steps=a.steps, seed=a.seed)
    G, _, hist = train(cfg, print)
    rng = np.random.default_rng(99)
    with open(a.svg, "w") as f:
        f.write(to_svg(sample_real(800, rng), G.forward(rng.normal(size=(800, cfg.z_dim)))))
    print(f"wrote {a.svg}; final: {hist[-1]}")


if __name__ == "__main__":
    main()
