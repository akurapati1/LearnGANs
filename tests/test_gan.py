import unittest
import xml.etree.ElementTree as ET

import numpy as np

from gan.data import sample_real
from gan.nn import MLP
from gan.train import Config, evaluate, to_svg, train


class GradientCheck(unittest.TestCase):
    def test_backward_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        net = MLP([3, 5, 4, 1], rng, out="sigmoid")
        x, y = rng.normal(size=(6, 3)), rng.integers(0, 2, (6, 1)).astype(float)

        def loss():
            p = np.clip(net.forward(x), 1e-9, 1 - 1e-9)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

        net.forward(x)
        grads, gx = net.backward((net.acts[-1] - y) / len(x))
        for pi, (p, g) in enumerate(zip(net.params(), grads)):
            idx = tuple(np.array([s // 2 for s in p.shape]))
            old = p[idx]; p[idx] = old + 1e-6; hi = loss(); p[idx] = old - 1e-6; lo = loss(); p[idx] = old
            self.assertAlmostEqual(g[idx], (hi - lo) / 2e-6, places=5, msg=f"param {pi}")
        old = x[0, 0]; x[0, 0] = old + 1e-6; hi = loss(); x[0, 0] = old - 1e-6; lo = loss(); x[0, 0] = old
        net.forward(x)
        self.assertAlmostEqual(net.backward((net.acts[-1] - y) / len(x))[1][0, 0], (hi - lo) / 2e-6, places=5)


class Metrics(unittest.TestCase):
    def test_real_data_scores_perfectly_and_noise_does_not(self):
        rng = np.random.default_rng(1)
        real = evaluate(sample_real(4000, rng))
        self.assertEqual(real["modes_covered"], 8)
        self.assertGreater(real["high_quality"], 0.95)
        blob = evaluate(rng.normal(0, 0.1, (4000, 2)))
        self.assertLessEqual(blob["modes_covered"], 0)
        self.assertLess(blob["high_quality"], 0.05)


class Training(unittest.TestCase):
    def test_generator_learns_the_ring(self):
        cfg = Config(steps=2500, seed=0)
        rng = np.random.default_rng(7)
        before = evaluate(MLP([cfg.z_dim, cfg.hidden, cfg.hidden, 2], np.random.default_rng(0)).forward(
            rng.normal(size=(2000, cfg.z_dim))))
        G, _, hist = train(cfg)
        after = evaluate(G.forward(rng.normal(size=(2000, cfg.z_dim))))
        self.assertGreater(after["high_quality"], before["high_quality"] + 0.3)
        self.assertGreaterEqual(after["modes_covered"], 6)

    def test_svg_is_valid(self):
        rng = np.random.default_rng(0)
        root = ET.fromstring(to_svg(sample_real(10, rng), sample_real(10, rng)))
        self.assertEqual(len(root.findall("{http://www.w3.org/2000/svg}circle")), 20)


if __name__ == "__main__":
    unittest.main()
