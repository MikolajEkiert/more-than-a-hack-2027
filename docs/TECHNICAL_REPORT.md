# culers: frame-level ball possession

More than a Hack 2027, phase 1. Team `culers`.
Repository: https://github.com/MikolajEkiert/more-than-a-hack-2027

| | |
|---|---|
| Submitted configuration | `configs/hybrid_supervised.toml` |
| sha256 of `culers.csv` | `25f146c056a7f76b73ad9c5a35b85066930b44ba675715133681bc7b5e70a298` |
| Regenerate | `make submit CONFIG=configs/hybrid_supervised.toml TEAM=culers` (section 7) |

Every number below comes from `docs/DECISIONS.md` and the run files in `experiments/runs/`.

## 1. Approach

**The question we optimised for.** The ranking counts whole possessions (`spellF1`), so a system
that is right on most frames but flickers scores badly. We therefore split the problem in three
stages with a fixed contract, and put all the possession structure in the last one:

```
tracking -> is the ball in play? -> which team has it? -> decoder -> H / A / D per frame
```

Each stage is a set of interchangeable methods (`src/possession/stages/`), a configuration file
picks one per stage, and every version is scored the same way. Two tracks were built on it: one
with no labels at all, one with a model trained on other matches. The submission combines them.

**What is submitted.**

1. **Evidence from a trained model** (`solution/`, reached through `src/possession/supervised.py`).
   One LightGBM with three classes, H / A / D, per frame. It sees the tracking only: 203 hand-made
   features (ball kinematics, distances from players to the ball, which team's player is nearest
   and for how long, touches detected from the ball changing direction or speed next to a player,
   restart layouts, how long the ball has been missing and when it is next seen), each also as
   averages over past, future and centred windows. It is trained on seven open Bundesliga matches
   (section 3) whose clean tracking is first degraded to look like automatic tracking: the ball
   removed in bursts, players dropping out, slow position drift. How much is removed is set from
   three unlabelled statistics of the competition tracking (40.7% of graded frames without a
   ball, mean missing stretch 17.2 s, 93% of players visible). No label of the provided match is
   involved.
2. **A kick-off rule from the formation** (`ball_state.kickoff_wait`). When each team stands
   entirely in its own half, play is waiting for a kick-off and is dead until the ball leaves the
   centre. This also labels the wait before the second half as dead. It uses no constant of this
   match; it lands 0.2 s from the kick-off time the organisers publish.
3. **A decoder that produces possessions, not frames** (`decode.switch_cost`). The most likely
   H / A / D sequence per half, where changing state has a price: 5 for a change of team, 12 to
   stop play and 12 to restart it (in units of log-probability, with the per-frame evidence
   divided by 5 because neighbouring frames repeat the same information). A possession therefore
   exists only if the evidence for it outweighs its two edges. The edges of stoppages are then
   moved onto the edges of the stretch without a ball they belong to.

**Why this and not something else.** Mean `spellF1` on matches the model was not trained on
(section 3 says what each set is):

| Version | Metrica, easier noise | Metrica, noise fitted to the match | IDSSE, out of fold | PFF, real noise |
|---|---|---|---|---|
| Organisers' baseline (`baseline_smooth50`) | 0.334 | 0.213 | | |
| Our pipeline without any labels (`unsup_v2`) | 0.598 | 0.394 | 0.416 | 0.686 |
| The trained model with its own decoder | 0.621 | not measured | 0.637 | not measured |
| **Submitted: the model's evidence, our kick-off rule, our decoder** | **0.659** | **0.524** | **0.671** | **0.773** |

The model names the right team more often than our label-free evidence; stoppage detection is
equal in both tracks on the Metrica matches. Our decoder adds 0.04 on the same evidence on the
Metrica matches and 0.03 on the IDSSE matches, on all seven of them, although it was never tuned
there and the model's own decoder was. On the competition match the model's most likely class
per frame makes 1,385 possessions; after decoding there are 300.

A change to the submitted configuration needs, by a rule written down before measuring: at least
0.02 better on the Metrica matches under both noise profiles, not worse under either reading of
"possession" (section 4), better on two of the three matches, and not worse on the IDSSE matches.
Nothing measured after the hybrid has passed it (section 5).

## 2. Assets used and dropped

| Asset | Used? | How, or why not |
|---|---|---|
| Tactical tracking, players and ball | yes | Positions only. The file's ball speed column is filled on 7,377 of 94,214 tracked graded frames and disagrees with the positions, so ball speed is computed from positions |
| Smart Tagging | for one audit, not in the method | 68% of graded frames lie in a phase that names a team. The team implied by the tag agrees with our output on 77.5% of those frames when the ball is tracked and on 47.7% when it is not, which is where help would be needed. Half of the frames we label dead lie inside such a phase. No other match has tags, so nothing based on them could be validated. Used once, as a check: the six kick-offs it tags are the six kick-off set-ups our formation rule finds, one to one |
| Homography | no | The tactical tracking is already in pitch coordinates |
| Tactical video | no | We built no detector or tracker of our own |
| Broadcast video and its tracking | no | Not evaluated. That recording is not synchronised to the tactical one and comes without a mapping (DATA.md), and its homography is usable on 65.8% of its frames. Aligning the two would be a project of its own, and no other match has two recordings, so we could not have validated what it added |
| Public 5-minute ground truth | check only | Printed next to every run to catch a broken pipeline. With 17 possessions its score has a 5-95% interval of 0.41 to 0.78 for our submission, so it cannot choose between versions and was not used to |
| Published full-match scores of the four baselines | for one validation profile, until 2026-11-14 | That profile (`published_baselines`) is fitted to numbers computed from the private ground truth, so since 2026-11-14 it decides nothing and is only reported. Nothing in the submitted configuration was accepted on its strength alone. Disclosure: before that date it tipped one decision, the rejection of adding our label-free evidence to the model's. That addition was then measured under a profile fitted without the ground truth: it gains 0.022 under the easier profile, loses 0.021 under the fitted one and loses 0.012 on the IDSSE matches, so it stays out, now on allowed evidence |

Tried and dropped, each measured on other matches: estimating stoppage cues from the match itself
(worse); giving the team of the last touch possession until the next touch (creates false short
possessions); a prior that the other team restarts after a stoppage (it does in 47% of cases);
the model's own boundary detector inside our decoder (no gain); adding our label-free evidence to
the model's (gains under one noise profile, loses under the other); training the model on 48 PFF
matches as well (+0.005 under the easier profile, -0.021 under the fitted one) or on those alone
(-0.016 and -0.082); training labels with short interruptions merged, with a slower decoder
(section 5); shifting possession changes in time (they are a median of 5 frames late, but even
perfect timing of the changes already found adds at most 0.03).

## 3. Training and validation data
No label of the provided match was used for training or for choosing anything, and we labelled
nothing by hand on any match.

| dataset | source and licence | labels: who and how | amount | used for |
|---|---|---|---|---|
| IDSSE / Sportec Open DFL Tracking and Event Data: 7 matches (2 Bundesliga, 5 2. Bundesliga, season 2022/23) | Bassek et al., Scientific Data 12:195 (2025), doi:10.1038/s41597-025-04505-y; CC-BY 4.0. Downloaded at run time from https://huggingface.co/datasets/pysport/idsse-data at the fixed revision `a715a38` (2.63 GB), every file checked against `solution/idsse.sha256`; not redistributed | Nobody on our side. The provider's own per-frame flags on corrected tracking (`solution/idsse.py`): `BallStatus` gives D where the ball is out of play, `BallPossession` gives H or A while it is in play | 7 matches, about 1.0 million frames at 25 Hz. Each match is used with home and away swapped as well, in two noise draws, every 5th frame | training the LightGBM of `solution/` (the only trained component of the hybrid). Its tracking is first degraded to resemble the competition tracking; the amounts are calibrated on unlabelled statistics of that tracking. Also validation, strictly out of fold: each match is scored by a model trained on the other six |
| Metrica Sports sample data, Sample Games 1-3 | https://github.com/metrica-sports/sample-data (public; no licence file, the README asks for credit; downloaded at run time, not redistributed) | Nobody on our side. The provider's corrected tracking and event files, turned into per-frame H/A/D by `src/possession/validation.py`: D where the tracking has no ball position and the events confirm a stoppage, otherwise the team in possession by the events, under two readings (`touch`, `controlled`) that are both scored | 3 matches, 427,071 graded frames; 1,059 possessions under `touch` (mean 10.2 s), 732 under `controlled` (mean 14.8 s); 32-40% dead | validation and the choice of hyperparameters only; nothing is trained on it |
| PFF FC FIFA World Cup 2022: 64 matches | PFF FC (Gradient Sports). Free, on request through the provider's form (https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset); not redistributed and not downloaded by our code | The provider's analysts: an event stream merged into the tracking frames (on-ball events with the team in possession; ball out; end of period). `solution/pff.py` carries the owner forward between events and marks the ball dead after an out or an end event | 64 matches of broadcast tracking at 29.97 Hz, resampled to 25 Hz. Possession changes hands 4.7 times per live minute, mean 13.1 s; no ball position on 44% of frames | validation only: 16 matches, fixed before measuring (sorted by id, every fourth), as a test set whose tracking noise is real. Training on the other 48 was measured and not adopted, so the submitted model has seen none of the 64 |

**How validation works.** Three sets of other matches, each with a role fixed in advance
(docs/FUSION.md section 8).

- **Metrica sample games: the evidence.** Same provider as the competition tracking. Their
  tracking is clean, so before scoring it is degraded to resemble ours, with three random seeds,
  under two profiles:
  - `label_free`: set from aggregates of our own tracking, without labels or scores. The
    organisers' baseline scores 0.334 on it against 0.2433 on the real match, so it is too easy.
  - `match_like`: fitted so that our two pipelines agree with each other on the degraded matches
    as they do on the competition match (possessions of one scored against the other: 0.456 on
    the match, 0.478 under the profile; same team where both say play: 0.829 and 0.814). No
    label and no published score is used. The baseline scores 0.213 on it.
- **IDSSE, out of fold: a veto.** Another provider, and the only labelled data we have with the
  ball in view during stoppages. Scored under two readings of the provider's flags whose
  possession density matches the two Metrica readings.
- **PFF test matches: a veto.** Real broadcast tracking, used as delivered: the ball has no
  position on 45% of frames, in stretches that are almost all longer than 3 s, like ours. Its
  players are mostly estimated positions, unlike ours, and our two pipelines agree with each
  other far more there than on the competition match (0.685 against 0.456), so it is an easier
  tracking than ours.
- **`published_baselines`: reported only.** A third profile, fitted so that the four organiser
  baselines score what EVALUATION.md publishes for the full match. Those numbers derive from the
  private ground truth; see section 2 for what the profile was used for and when it stopped.

## 4. Possession rules implemented

**Two readings of "possession", both always scored.** The other matches' labels come from events,
and the definition can be read two ways: any touch changes the owner (`touch`, about 6.0
possessions per minute of play), or a touch counts only if the team then keeps the ball for 0.2 s
or completes a pass, shot or set piece (`controlled`, about 4.05). We could not tell which the
private ground truth follows, so nothing was accepted that loses under either.

- **Change of possession.** The team changes only when the evidence for the other team outweighs
  the price of a change. A single touch, a deflection or a header straight back does not carry
  enough; a sequence of frames with the other team in control does. The change is placed on the
  first frame that carries evidence for the new team.
- **Loose and contested balls, and play while the ball is not tracked.** Frames without evidence
  for either team cost both the same, so possession stays with the team that had it, as
  CHALLENGE.md asks.
- **Dead ball.** A stoppage needs evidence that outweighs stopping and restarting (24 in total),
  so short dips in the evidence do not create one. A stoppage that begins within 3 s after the
  ball was lost is dated from the moment it was lost; a restart is dated from the moment the ball
  reappears if it is then moving at 3 m/s or more.
- **Restarts.** The team after a stoppage is whatever the evidence says at the restart; no rule
  ties it to the team before. Measured on other matches, it is the other team in 47% of cases.
- **Kick-offs and the start of the second half.** Dead until the kick, by the formation rule of
  section 1.
- **Abstention.** We never write `X` inside a graded half. `X` is written outside the halves
  only, where nothing is scored.

## 5. Results

**The submitted configuration on matches the model was not trained on** (`spellF1`):

| | Metrica, easier noise | Metrica, noise fitted to the match | IDSSE, out of fold | PFF, real noise |
|---|---|---|---|---|
| mean | 0.659 | 0.524 | 0.671 | 0.773 |
| by reading | 0.643 / 0.675 | 0.518 / 0.531 | 0.690 / 0.651 | one reading |
| lowest and highest match | 0.635, 0.673 (3) | 0.511, 0.537 (3) | 0.646, 0.717 (7) | 0.698, 0.817 (16) |
| team accuracy where both say play | 0.923 | 0.877 | 0.901 | 0.972 |
| play / stoppage F1 | 0.936 | 0.935 | 0.970 | 0.940 |

Runs: `experiments/runs/20261114T012556*`, `T154740*`, `T152121*`, `T160459*`. The spread over
degradation seeds is 0.007.

**The choice between two versions.** Besides the submitted configuration (version 1) we built a
second one that bets on how the ground truth reads "possession" (version 3: a model trained on
labels with interruptions shorter than 2 s merged, and a decoder that is slower to change team).
The rule for choosing was written down before measuring (docs/FUSION.md section 8). Version 3
minus version 1:

| Set | Difference | Rule |
|---|---|---|
| Metrica, easier noise, `touch` / `controlled` | -0.030 / +0.042 | evidence: mean at least +0.02, no scenario below -0.02 |
| Metrica, fitted noise, `touch` / `controlled` | +0.011 / +0.072 | |
| mean of the four | +0.024 (+0.019 when the variant is chosen on two matches and scored on the third) | fails: one scenario is at -0.030 |
| IDSSE, out of fold | +0.020, 7 of 7 matches | veto: passes |
| PFF test matches | -0.052, 0 of 16 matches | veto: fails |

Version 1 is submitted. Version 3 fails the rule twice: it loses 0.030 in one evidence scenario,
and on the PFF matches, where the tracking noise is real and the provider's labels change hands
4.7 times per live minute (the public window: 4.2), it loses on every match. On the competition
match it would make 223 possessions where version 1 makes 300. Run:
`experiments/runs/20261114T160229*`; the family it was picked from:
`scripts/variant_grid.py family label_free match_like`.

Adding our label-free evidence to the model's at half weight, the one change an earlier profile
had rejected alone (section 2), was measured again: +0.022 under the easier profile, -0.021
under the fitted one, -0.012 on the IDSSE matches (0 of 7), +0.009 on the PFF matches. It stays
out. Runs: `experiments/runs/20261114T154049*`, `T154339*`, `T161044*`, `T161937*`.

**Where the rest is.** Replacing one input by the truth on the Metrica matches (the harsher
published profile): perfect stoppages give 0.635, a perfect team on frames without a ball 0.640,
a perfect team on frames with a ball 0.786, everything perfect 0.965. The decoder is not what
limits the result; recognising the team is. In the first two seconds after a true change of
possession the model names the right team on 62-67% of frames, against 88-94% elsewhere. True
possessions shorter than 5 s are recovered in 15-41% of cases, those longer than 15 s in 86%.

**On the competition match, without labels:** 300 possessions, mean 10.7 s, 49% of graded frames
dead, home share of possession 60%. Run: `experiments/runs/20261114T012451*`. The six kick-off
set-ups the formation rule finds are the six kick-offs in Smart Tagging.

**Public 5 minutes, as a check only:** `spellF1` 0.594, 16 possessions predicted against 17 true.

**What we expect.** Nearer the figure under the noise fitted to the match (0.52) than the others.
The reason is label-free: our two pipelines agree with each other on 0.456 of possessions on the
competition match, against 0.63 under the easier profile and 0.685 on the PFF matches, so both
of those are easier tracking than ours; the fitted profile was made to reproduce that agreement
(0.478). These figures rank our own versions; they are not a forecast: the noise is simulated
where it is not another provider's, and the reading of the labels is unknown.

## 6. Limitations

- **We do not know which reading of possession the ground truth follows.** The decoder's price
  for a change of team (5) is the value that does not lose under either reading on the Metrica
  matches. On the PFF matches it scores 0.773, against 0.772 at a price of 3 and 0.755 at 8, and
  predicts 0.98 possessions per true one. A sparser version gains under one reading and loses
  under the other.
- **The noise is simulated on the matches that share our provider.** We cannot know which of our
  ball gaps are stoppages, so the degradation of the Metrica matches is a model of the tracker,
  fitted to label-free statistics. The matches with real noise (PFF) are another provider's.
- **Stoppages with the ball in view are validated only on other providers' data.** We label 12.6%
  of the match's frames dead while the ball is tracked. On the IDSSE matches 1% of true stoppage
  frames with the ball in view are called play; on the PFF matches 18%; the Metrica matches
  cannot test it, because their clean tracking has no ball during stoppages.
- **Short possessions and the moments of change are where we lose.** With more time we would work
  on the team stage. More training matches of another provider did not help on the Metrica
  matches (section 2), so the next step would be features aimed at the first seconds after a
  change.
- **Few validation matches of our provider** (three), and many comparisons made on them. That is
  why the two other sets were added as vetoes.
- **The model is trained on seven matches** of another league and another tracking system.
- **We did not use the video.**

## 7. Reproduction

```bash
uv sync                                   # Python 3.11, every dependency pinned by uv.lock
# unpack the data bundle into ./data as DATA.md describes (or set MTAH_DATA_DIR)
make submit CONFIG=configs/hybrid_supervised.toml TEAM=culers
# without make: uv run possession submit configs/hybrid_supervised.toml --team culers
```

This writes `submission/culers.csv`, runs the pipeline a second time from a fresh parse of the
data, requires the same sha256, calls the organisers' validator and prints the sha256 above. It
takes under a minute.

- **The trained model is in the repository** (`models/`, 2.7 MB, with its sha256). It is stored
  under the cache key the training code computes, so it is used only if it is exactly the model
  this code would train; nothing has to be downloaded.
- **To retrain it from scratch instead:** `MTAH_NO_CACHE=1 uv run possession run
  configs/hybrid_supervised.toml`. This downloads the open training data (2.63 GB, fixed revision,
  every file checked by hash) and takes about four minutes. The retrained model file is
  byte-identical to the committed one.
- **Checked on:** macOS arm64, a fresh clone there, a Linux arm64 container and a Linux x86-64
  container under emulation: the same file each time. Not checked: Windows.
- **Not the submission path:** `python -m solution.submission` and the notebooks in `notebooks/`
  are the supervised track's own experiments. They use another decoder and produce another file.
