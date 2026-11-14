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

**Why this and not something else.** Mean `spellF1` on three other matches (section 3), each
under two noise profiles:

| Version | easier profile | harsher profile |
|---|---|---|
| Organisers' baseline (`baseline_smooth50`) | 0.334 | 0.238 |
| Our pipeline without any labels (`unsup_v2`) | 0.598 | 0.448 |
| The trained model with its own decoder | 0.621 | 0.506 |
| **Submitted: the model's evidence, our kick-off rule, our decoder** | **0.659** | **0.561** |

The model names the right team more often than our label-free evidence (team accuracy 0.92 and
0.89 against 0.89 and 0.83); stoppage detection is equal in both tracks. Our decoder adds 0.04 to
0.05 on the same evidence. On the competition match the model's most likely class per frame makes
1,385 possessions; after decoding there are 300.

A change was accepted only by a rule written down before measuring: at least 0.02 better on
average, not worse under either reading of "possession" (section 4), better on at least two of the
three matches, and all of that under both noise profiles.

## 2. Assets used and dropped

| Asset | Used? | How, or why not |
|---|---|---|
| Tactical tracking, players and ball | yes | Positions only. The file's ball speed column is filled on 7,377 of 94,214 tracked graded frames and disagrees with the positions, so ball speed is computed from positions |
| Smart Tagging | no, after measuring | 68% of graded frames lie in a phase that names a team. The team implied by the tag agrees with our output on 77.5% of those frames when the ball is tracked and on 47.7% when it is not, which is where help would be needed. Half of the frames we label dead lie inside such a phase. No other match has tags, so nothing based on them could be validated |
| Homography | no | The tactical tracking is already in pitch coordinates |
| Tactical video | no | We built no detector or tracker of our own |
| Broadcast video and its tracking | no | Not evaluated |
| Public 5-minute ground truth | check only | Printed next to every run to catch a broken pipeline. With 17 possessions its score has a 5-95% interval of 0.41 to 0.78 for our submission, so it cannot choose between versions and was not used to |
| Published full-match scores of the four baselines | for one validation profile | See section 3. Nothing was accepted on its strength alone. It did decide one rejection: adding our label-free evidence to the model's gained 0.022 under the easier profile and was not adopted because it lost 0.007 under this one |

Tried and dropped, each measured on other matches: estimating stoppage cues from the match itself
(worse); giving the team of the last touch possession until the next touch (creates false short
possessions); a prior that the other team restarts after a stoppage (it does in 47% of cases);
the model's own boundary detector inside our decoder (no gain); adding our label-free evidence to
the model's (gains under one noise profile, loses under the other); training labels with short
interruptions merged, with a slower decoder (gains under one reading of possession, loses under
the other); shifting possession changes in time (they are a median of 5 frames late, but even
perfect timing of the changes already found adds at most 0.03).

## 3. Training and validation data
No label of the provided match was used for training or for choosing anything, and we labelled
nothing by hand on any match.

| dataset | source and licence | labels: who and how | amount | used for |
|---|---|---|---|---|
| IDSSE / Sportec Open DFL Tracking and Event Data: 7 matches (2 Bundesliga, 5 2. Bundesliga, season 2022/23) | Bassek et al., Scientific Data 12:195 (2025), doi:10.1038/s41597-025-04505-y; CC-BY 4.0. Downloaded at run time from https://huggingface.co/datasets/pysport/idsse-data at the fixed revision `a715a38` (2.63 GB), every file checked against `solution/idsse.sha256`; not redistributed | Nobody on our side. The provider's own per-frame flags on corrected tracking (`solution/idsse.py`): `BallStatus` gives D where the ball is out of play, `BallPossession` gives H or A while it is in play | 7 matches, about 1.0 million frames at 25 Hz. Each match is used with home and away swapped as well, in two noise draws, every 5th frame | training the LightGBM of `solution/` (the only trained component of the hybrid). Its tracking is first degraded to resemble the competition tracking; the amounts are calibrated on unlabelled statistics of that tracking. Not used by our validation |
| Metrica Sports sample data, Sample Games 1-3 | https://github.com/metrica-sports/sample-data (public; no licence file, the README asks for credit; downloaded at run time, not redistributed) | Nobody on our side. The provider's corrected tracking and event files, turned into per-frame H/A/D by `src/possession/validation.py`: D where the tracking has no ball position and the events confirm a stoppage, otherwise the team in possession by the events, under two readings (`touch`, `controlled`) that are both scored | 3 matches, 427,071 graded frames; 1,059 possessions under `touch` (mean 10.2 s), 732 under `controlled` (mean 14.8 s); 32-40% dead | validation and the choice of hyperparameters only; nothing is trained on it |

**How validation works.** The three Metrica matches are clean. Before scoring, their tracking is
degraded to resemble ours, with three random seeds, under two profiles kept side by side:

- `label_free`: set from aggregates of our own tracking, without labels or scores. The
  organisers' baseline scores 0.334 on it against 0.2433 on the real match, so it is too easy.
- `published_baselines`: fitted so that the four organiser baselines score what EVALUATION.md
  publishes for the full match. Those eight numbers are computed from the private ground truth,
  so we state plainly what this profile was used for: it is a second, harsher test that a change
  also had to pass. Everything in the submitted configuration was accepted under the easier
  profile as well; the one thing this profile decided alone was a rejection (section 2).
  Label-free statistics of the match (how often our two tracks agree) are closer to this profile
  than to the easier one.

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

**On other matches** (three matches, three seeds each, both readings; `spellF1`):

| | easier profile | harsher profile |
|---|---|---|
| Submitted configuration, mean | 0.659 | 0.561 |
| under `touch` / `controlled` | 0.643 / 0.675 | 0.546 / 0.576 |
| per match | 0.673, 0.669, 0.635 | 0.561, 0.558, 0.564 |
| team accuracy where both say play | 0.923 | 0.896 |
| play / stoppage F1 | 0.936 | 0.928 |

Runs: `experiments/runs/20261114T012556*` and `20261114T012701*`. The spread over seeds is
0.007. The model was trained on none of these matches.

**Where the rest is.** Replacing one input by the truth on those matches (harsher profile): perfect
stoppages give 0.635, a perfect team on frames without a ball 0.640, a perfect team on frames with
a ball 0.786, everything perfect 0.965. The decoder is not what limits the result; recognising the
team is. In the first two seconds after a true change of possession the model names the right team
on 62-67% of frames, against 88-94% elsewhere. True possessions shorter than 5 s are recovered in
15-41% of cases, those longer than 15 s in 86%.

**On the competition match, without labels:** 300 possessions, mean 10.7 s, 49% of graded frames
dead, home share of possession 60%. Run: `experiments/runs/20261114T012451*`.

**Public 5 minutes, as a check only:** `spellF1` 0.594, 16 possessions predicted against 17 true.

**What we expect.** Around the harsher profile's figure, not the easier one. That reading rests
on label-free statistics of the match, not on how that profile was fitted: our two tracks agree
with each other at 0.46 on the match, 0.49 under the harsher profile and 0.63 under the easier
one, and their team agreement is 0.83, 0.82 and 0.89 (docs/DECISIONS.md, 2026-11-14). This is a ranking
tool between our own versions, not a forecast: the noise is simulated and the reading of the
labels is unknown.

## 6. Limitations

- **We do not know which reading of possession the ground truth follows.** The decoder's price
  for a change of team (5) is the value that does not lose under either reading; a value of 8
  would be worth about 0.03 more under one and cost 0.02 under the other.
- **The noise is simulated.** We cannot know which of our ball gaps are stoppages, so the
  degradation of other matches is a model of the tracker, calibrated on aggregates.
- **Stoppages with the ball in view are not validated.** We label 12.6% of the match's frames
  dead while the ball is tracked; in our degraded test matches such frames are 2-4%, because
  their clean tracking has no ball during stoppages.
- **Short possessions and the moments of change are where we lose.** See section 5. With more
  time we would work on the team stage: more training matches with real tracking noise, and
  features aimed at the first seconds after a change.
- **Three validation matches of one provider**, and many comparisons made on them.
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
