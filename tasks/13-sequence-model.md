---
status: review
owner: MikolajEkiert (Claude)
stage: ball_state, team, infra
---

# 13 · A temporal network over the package's features

## Goal
Members M5 and M6 of the family in docs/FUSION.md section 10: ball-state evidence from a network
that sees about 51 s around each frame, trained in the mixed ball regime chosen by gate G2.

## Fixed before any network was trained
- Architecture (`sequence.ARCH`): 5 Hz input (every 5th frame of each half), the package's 203
  features standardised and clipped to +-5, NaN as 0, plus a missing flag for every feature NaN on
  more than 5% of training rows; 1x1 projection to 64 channels with ReLU; 7 residual blocks
  `h += ReLU(conv3, dilation d)`, d = 1, 2, 4, ..., 64 (receptive field 255 steps, about 51 s);
  two 1x1 heads, `z_dead` and `z_home`. Zero padding at the edges of each half.
- Training (`sequence_train.TRAIN`): 2-minute crops at random offsets, batch 16, 30 epochs,
  AdamW (lr 2e-3, weight decay 1e-4, one-cycle schedule), gradient clipping 1.0, seed 0, 8 CPU
  threads, deterministic algorithms. Loss: binary cross-entropy of `z_dead` on graded frames and of
  `z_home` on live frames, both weighted by the package's sample weights. No early stopping, no
  choice of epochs or settings on any evaluation set.
- Data: the 7 IDSSE matches, both orientations; noise seed 0 in the package's degradation and seed
  1 in the automatic ball regime `auto_lf` (the mixed regime `mix_lf`). Out of fold: per noise
  seed, a network on the other six matches in that seed's regimes predicts the held-out match.
- Inference: numpy float64 (`sequence.py`), no torch at runtime; every export must reproduce
  torch's float64 output on a stored input to 1e-9 (gate G4).

## Interface
- `ball_state.sequence`, `team.sequence`; configs `hybrid_sequence_blend.toml` (M5) and
  `hybrid_sequence.toml` (M6).
- `uv run --with torch==2.15.0 possession train-sequence --folds --final`.

## Acceptance criteria
- [x] numpy inference tested against a naive convolution; importable without torch
- [x] G4 on every exported network (final: 9e-15)
- [x] M5 and M6 measured with scripts/family_grid.py; decision by the rule (version 1 stays)

## Log
- 2026-11-14: 14 fold networks and the final one trained on the CPU (1 h 22 min); G4 9e-15.
- 2026-11-14: M6 +0.025 / +0.018 on Metrica (held out), -0.001 IDSSE (3/7), +0.044 PFF (16/16).
  The rule keeps version 1 (+0.018 < +0.02 under match_like; IDSSE veto).
