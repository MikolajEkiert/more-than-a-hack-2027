import subprocess
import sys

import numpy as np

from possession import sequence
from possession.stages.ball_state import sequence as sequence_ball_state
from possession.stages.team import sequence as sequence_team
from tests.conftest import make_match


def _naive_conv(x, w, b, dilation):
    steps, cin = x.shape
    taps, _, cout = w.shape
    pad = dilation * (taps - 1) // 2
    out = np.zeros((steps, cout))
    for t in range(steps):
        for o in range(cout):
            acc = b[o]
            for k in range(taps):
                s = t + k * dilation - pad
                if 0 <= s < steps:
                    acc += sum(x[s, i] * w[k, i, o] for i in range(cin))
            out[t, o] = acc
    return out


def test_conv_same_matches_a_naive_reference():
    rng = np.random.default_rng(0)
    x, w, b = rng.normal(size=(40, 3)), rng.normal(size=(3, 3, 4)), rng.normal(size=4)
    for dilation in (1, 2, 7):
        np.testing.assert_allclose(
            sequence.conv_same(x, w, b, dilation), _naive_conv(x, w, b, dilation), atol=1e-12
        )
    one = rng.normal(size=(1, 3, 2))
    np.testing.assert_allclose(sequence.conv_same(x, one, b[:2], 1), x @ one[0] + b[:2], atol=1e-12)


def _tiny_net(n_in, seed=1):
    rng = np.random.default_rng(seed)
    width = 4
    net = {
        "ds": 5,
        "clip": 5.0,
        "dilations": (1, 2),
        "mean": np.zeros(n_in),
        "std": np.ones(n_in),
        "flag": np.array([0]),
        "proj_w": rng.normal(size=(1, n_in + 1, width)),
        "proj_b": rng.normal(size=width),
        "head_w": rng.normal(size=(1, width, 2)),
        "head_b": rng.normal(size=2),
    }
    for j in range(2):
        net[f"w{j}"] = rng.normal(size=(3, width, width)) * 0.3
        net[f"b{j}"] = rng.normal(size=width) * 0.1
    return net


def test_scale_clips_fills_and_flags():
    net = _tiny_net(2)
    net["mean"], net["std"] = np.array([1.0, 0.0]), np.array([2.0, 1.0])
    z = sequence.scale(np.array([[3.0, 100.0], [np.nan, -1.0]]), net)
    np.testing.assert_allclose(z, [[1.0, 5.0, 0.0], [0.0, -1.0, 1.0]])


def test_logits_run_each_half_alone_and_are_zero_outside():
    n = 400
    half = np.zeros(n, dtype=np.int8)
    half[20:200], half[220:390] = 1, 2
    rng = np.random.default_rng(2)
    x = rng.normal(size=(n, 3))
    net = _tiny_net(3)
    out = sequence.logits(net, x, half)
    assert out.shape == (n, 2) and np.all(out[half == 0] == 0)
    changed = x.copy()
    changed[220:390] += 3.0  # the second half only
    again = sequence.logits(net, changed, half)
    np.testing.assert_array_equal(again[half == 1], out[half == 1])
    assert not np.allclose(again[half == 2], out[half == 2])


def test_the_inference_module_never_needs_torch():
    # a fresh interpreter, so that reloading never touches this process's module (its `final`
    # cache, or the predictions the autouse fixture stubs)
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "import numpy as np\n"
        "from possession import sequence\n"
        "assert sequence.conv_same(np.ones((3, 1)), np.ones((1, 1, 1)), np.zeros(1), 1).shape"
        " == (3, 1)\n"
        "assert sys.modules['torch'] is None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_stage_methods_read_the_two_heads_with_the_right_signs(monkeypatch):
    match = make_match(n=600)
    z = np.zeros((match.n_frames, 2))
    z[:300, 0], z[300:, 0] = 6.0, -6.0  # z_dead: dead, then live
    z[::2, 1], z[1::2, 1] = 6.0, -6.0  # z_home: home, then away
    monkeypatch.setattr(sequence, "predictions", lambda m: z)

    alive, home = sequence_ball_state(match), sequence_team(match)
    np.testing.assert_allclose(alive, 1.0 / (1.0 + np.exp(z[:, 0])))
    np.testing.assert_allclose(home, 1.0 / (1.0 + np.exp(-z[:, 1])))
    assert alive[:300].max() < 0.01 and alive[300:].min() > 0.99
    assert home[::2].min() > 0.99 and home[1::2].max() < 0.01


def test_the_committed_network_matches_its_listing_and_torch():
    listing = (sequence.MODELS_DIR / "sha256").read_text().splitlines()
    listed = {line.split()[1] for line in listing if line}
    assert listed == {p.name for p in sequence.MODELS_DIR.glob("*.npz")}
    net = sequence.final()  # raises if the sha256 does not match the listing
    assert sequence.check(net) < 1e-9  # gate G4: numpy reproduces torch's float64 output
