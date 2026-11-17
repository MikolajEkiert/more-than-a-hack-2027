# culers: frame-level ball possession

More than a Hack 2027, phase 1 · team `culers` · https://github.com/MikolajEkiert/more-than-a-hack-2027

| | |
|---|---|
| Submitted configuration | `configs/c14_best.toml` (version 4) |
| `culers.csv` sha256 | `6fd87cc4467530fa3b208a9dc9458d91949a8c98ef8a1b6ffc69cf9622fe29b7` |
| Regenerate | `make submit CONFIG=configs/c14_best.toml TEAM=culers` (under a minute) |
| Full report | `docs/TECHNICAL_REPORT.md`: every measurement, rule and tried idea |

## 1. Approach

`spellF1` scores whole possessions, so flicker is fatal and per-frame accuracy is not the goal. We
split the problem into three stages and put all the possession structure in the last one:

```
tracking -> is the ball in play? -> which team has it? -> decoder -> H / A / D per frame
```

1. **Is the ball in play: a temporal network.** A dilated convolutional network sees about 51 s
   around every frame at 5 Hz, over 203 per-frame features (ball kinematics, player-ball
   distances, touches, restart layouts, time since the ball was lost, team speeds). A stoppage is
   recognised from what happens around it, which a per-frame model cannot see. Inference is plain
   numpy in float64, so every machine gives the same numbers.
2. **Which team has it: LightGBM with our team features.** The 203 features plus 73 of our own,
   built for a ball that is tracked but noisy. For example: how many of each team's players are
   between the ball and their own goal, whether a team's nearest player moves with the ball, and
   how fast each team closes in on the ball.
3. **A kick-off rule from the formation.** When each team stands entirely in its own half, play
   waits for a kick-off and is dead until the ball leaves the centre. There is no constant of this
   match: the rule lands 0.2 s from the published kick-off time and finds the six kick-offs that
   Smart Tagging lists.
4. **A decoder that outputs possessions, not frames.** It takes the most likely H/A/D sequence per
   half, where each change of state has a price: 5 for a change of team, 16 to stop play, 12 to
   restart it. A possession exists only if the evidence for it outweighs its two edges.

## 2. Results on matches the models never saw

Mean `spellF1`, with our models and the baseline scored the same way:

| | Metrica, noise as hard as the match | Metrica, the match's ball regime | IDSSE, out of fold | PFF, real tracking noise (16 matches) |
|---|---|---|---|---|
| Organisers' baseline | 0.213 | | | |
| Version 1 (our earlier pipeline) | 0.524 | 0.655 | 0.671 | 0.773 |
| **Version 4 (submitted)** | **0.590** | **0.703** | **0.671** | **0.820** |
| Gain over version 1 [5-95%] | +0.066 [+0.054, +0.074] | +0.049 [+0.040, +0.055] | +0.001 [-0.007, +0.005] | +0.046 [+0.039, +0.055] |

- **How a version is accepted.** It must pass a decision rule written down before the
  measurement that decides. The rule chooses on two Metrica matches, scores on the third,
  bootstraps over (match, 10-minute window) blocks, and sets loss floors on the other sets.
  Version 4 wins on every held-out Metrica match and on 16 of 16 PFF matches.
- **Against other teams' public predictions.** On PFF, which none of us trained on: version 4
  scores 0.820, Barca-Hackathon 0.743, Team Sigue 0.767. Version 4 is higher on all 16 matches
  against each.
- **On the competition match, without labels.** 271 possessions (4.9 per live minute, mean
  12.2 s), and 49% of graded frames are dead.
- **Public 5 minutes, as a pipeline check only.** `spellF1` 0.723. It chose nothing.
- **Our rough expectation for the full match:** 0.59-0.70, the range of the two profiles fitted
  to the match. It is a guess from simulated noise, not a measurement.

## 3. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical tracking | **Used.** Positions only. Ball speed is recomputed from positions, because the file's speed column is mostly empty and disagrees with them. |
| Smart Tagging | **One audit only.** It confirmed the six kick-offs. No other match has tags, so nothing based on them could be validated. |
| Homography, tactical video, broadcast video and tracking | **Not used.** The tracking is already in pitch coordinates. The broadcast is not synchronised to the tactical recording, and no other match has two recordings to validate on. |
| Public 5-minute ground truth | **Check only.** 17 possessions give a 5-95% interval about 0.37 wide, too wide to choose between versions. |

**Tried and dropped (measured on other matches):**
- a minimum stoppage length;
- training the team model on PFF or SkillCorner as well;
- debounced or hand-over-weighted training labels;
- seed ensembles of the network;
- a restart-goes-to-the-other-team prior;
- last-touch possession;
- a decoder maximising expected `spellF1`;
- relabelling short sandwiched possessions.

Ideas taken from other teams are credited in `docs/DECISIONS.md`. None of them was adopted.

## 4. Training and validation data

No label of the provided match was used for training or choosing anything. We labelled nothing by
hand on any match.

| Dataset | Labels: who and how | Amount | Use |
|---|---|---|---|
| **IDSSE / Sportec Open DFL** (Bassek et al., Sci. Data 2025; CC-BY 4.0; downloaded at a pinned revision and hash-checked) | The provider's per-frame `BallStatus` gives D; `BallPossession` gives H or A | 7 matches, about 1.0 M frames, home/away-swapped copies, 2 noise draws | **Training of both models.** Tracking is degraded to resemble the competition tracking, with amounts calibrated on unlabelled statistics of that tracking. Validation out of fold. |
| **Metrica sample games 1-3** (public GitHub) | The provider's events turned into H/A/D under two readings: `touch` and `controlled` | 3 matches, 427 k graded frames | Validation only. Degraded under profiles fitted to the match's label-free statistics. |
| **PFF FC World Cup 2022** (free on request; not redistributed) | The provider's analysts' event stream merged into tracking | 16 test matches, fixed before measuring | Validation only. This is real broadcast tracking noise. |

## 5. Possession rules

- **Change of possession.** The team changes only when the evidence for the other team outweighs
  the price of a change. A single touch, a deflection or a header straight back does not change it.
  The change is placed on the first frame with evidence for the new team.
- **Loose balls, challenges and ball not tracked.** Frames without evidence for either team cost
  both the same, so possession stays with the team that had it.
- **Dead ball.** It needs evidence that outweighs stopping plus restarting, so short dips do not
  create a stoppage. A stoppage is dated from when the ball was lost (within 3 s) or left the
  pitch (within 6 s, Law 9). A restart is dated from when the ball reappears moving at 3 m/s or more.
- **Restarts.** The team at a restart comes from the evidence, with no rule tying it to the team
  before the stoppage.
- **Kick-offs.** The ball is dead until the kick, by the formation rule.
- **Abstention.** `X` is written only outside the graded halves.

Because the definition can be read two ways, every version is scored under both `touch`
(about 6.0 possessions per minute) and `controlled` (about 4.1). Version 4 gains under both.

## 6. Limitations

- **The ground truth's reading of "possession" is unknown.** A change price of 6 instead of 5
  would gain about 0.01 if it reads `controlled`, and loses on PFF.
- **The noise on our provider's matches is simulated.** Identity switches and duplicated
  players are not simulated.
- **A fifth of the match is outside the validation.** About 20% of its 10-minute windows have
  a stoppage density that no Metrica or IDSSE window reaches.
- **The validation and training data are small.** Three validation matches of our provider are
  used for many comparisons, which is why PFF and IDSSE vetoes sit beside them. Both models are
  trained on seven matches of another league.
- **The main remaining loss** is possessions shorter than 5 s and the first seconds after a
  change of possession. The video is not used.

## 7. Reproduction

```bash
uv sync                                    # Python 3.11, all dependencies pinned (uv.lock)
# unpack the data bundle into ./data as DATA.md describes (or set MTAH_DATA_DIR)
make submit CONFIG=configs/c14_best.toml TEAM=culers
```

**What `make submit` does.** It writes `submission/culers.csv`, then reruns the pipeline from a
fresh parse of the data and requires the same sha256. It then runs the organisers' validator.

**Trained models.** Both are committed (`models/team/`, `models/sequence/`) with their sha256, so
nothing is trained or downloaded. Retraining from scratch is documented and gives byte-identical
LightGBM with 8 threads.

**Where it was checked.** On macOS arm64, and from a fresh clone with an empty cache in a Linux arm64 container
and an emulated Linux x86-64 container. Every run gave the same file and the validator passed.
