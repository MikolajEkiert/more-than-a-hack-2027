# culers: frame-level ball possession

More than a Hack 2027, phase 1. Team `culers`.
Repository: https://github.com/MikolajEkiert/more-than-a-hack-2027

| | |
|---|---|
| Submitted configuration | `configs/c14_best.toml` (version 4) |
| sha256 of `culers.csv` | `6fd87cc4467530fa3b208a9dc9458d91949a8c98ef8a1b6ffc69cf9622fe29b7` |
| Regenerate | `make submit CONFIG=configs/c14_best.toml TEAM=culers` (section 7) |
| Produced from | commit `8517976` on `main` (2026-11-17); `make submit` ran the pipeline twice, the second time from a fresh parse, got the same sha256, and the official `evaluation/validate_submission.py` reports the file valid |

Every number below comes from `docs/DECISIONS.md`, docs/FUSION.md and the run files in
`experiments/runs/`. Version 1, the configuration we had ready before (`configs/hybrid_supervised.toml`,
sha256 `25f146c0...`), stays reproducible on the branch `fallback/hybrid`; section 5 says why we
moved on and which of our own rules would not have.

## 1. Approach

**The question we optimised for.** The ranking counts whole possessions (`spellF1`), so a system
that is right on most frames but flickers scores badly. We therefore split the problem in three
stages with a fixed contract, and put all the possession structure in the last one:

```
tracking -> is the ball in play? -> which team has it? -> decoder -> H / A / D per frame
```

Each stage is a set of interchangeable methods (`src/possession/stages/`), a configuration file
picks one per stage, and every version is scored the same way on other matches (section 3).

**What is submitted (version 4).**

1. **Is the ball in play: a temporal network** (`possession/sequence.py`, `ball_state.sequence`).
   A dilated convolutional network sees about 51 s around every frame, at 5 Hz, of the same 203
   per-frame features the LightGBM of the supervised package uses (ball kinematics, distances from
   players to the ball, touches, restart layouts, how long the ball has been missing, team speeds,
   each also averaged over windows). It is trained on seven open Bundesliga matches (section 3)
   degraded to look like automatic tracking, in two ways at once: one noise draw loses the ball at
   random, the other as automatic tracking does (during stoppages, apart from a few seconds at
   their start). Trained deterministically on the CPU; inference is plain numpy in float64, the
   same numbers on every machine. A stoppage is recognised from what happens around it, which a
   per-frame model cannot see.
2. **Which team has it: a LightGBM with our team features** (`possession/team_features.py`,
   `team.supervised` with `features = "v1+team"`). The package's model and training recipe, with
   73 features of our own appended to its 203, aimed at a ball that is tracked but noisy: how many
   of each team's players are between the ball and their own goal (the defending team has most of
   them; this hardly moves when the ball is 1-2 m off), whether each team's nearest player moves
   with the ball, the smallest per-player mean distance to the ball over 0.4 to 2.4 s (the
   possessor stays near it, a player passing by does not), how fast each team's players move
   toward the ball (defenders close in, the team in possession spreads), each team's spread
   against its own median, and past and future means of these. Its P(H) / (P(H) + P(A)) is the
   team evidence.
3. **A kick-off rule from the formation** (`ball_state.kickoff_wait`). When each team stands
   entirely in its own half, play is waiting for a kick-off and is dead until the ball leaves the
   centre. It uses no constant of this match; it lands 0.2 s from the kick-off time the
   organisers publish, and its six set-ups are the six kick-offs in Smart Tagging.
4. **A decoder that produces possessions, not frames** (`decode.switch_cost`). The most likely
   H / A / D sequence per half, where changing state has a price: 5 for a change of team, 16 to
   stop play and 12 to restart it (in units of log-probability, with the per-frame evidence
   divided by 5 because neighbouring frames repeat the same information). A possession therefore
   exists only if the evidence for it outweighs its two edges. The edges of stoppages are then
   moved onto the edges of the stretch without a ball they belong to, and a stoppage that begins
   within 6 s after the tracked ball left the pitch begins when it left (Law 9).

**Why this and not something else.** Mean `spellF1` on matches none of our models was trained on
(section 3 says what each set is):

| Version | Metrica, easier noise | Metrica, noise as hard as the match | Metrica, the match's ball regime | IDSSE, automatic ball, out of fold | IDSSE, out of fold | PFF, real noise |
|---|---|---|---|---|---|---|
| Organisers' baseline (`baseline_smooth50`) | 0.334 | 0.213 | | | | |
| Our pipeline without any labels (`unsup_v2`) | 0.598 | 0.394 | | | 0.416 | 0.686 |
| Version 1: the package's model, our kick-off rule and decoder | 0.659 | 0.524 | 0.655 | 0.665 | 0.671 | 0.773 |
| **Version 4 (submitted)** | **0.711** | **0.590** | **0.703** | **0.689** | **0.671** | **0.820** |

Each part was measured on its own before they were put together (docs/DECISIONS.md, tasks 13-14):
the temporal network for stoppages adds about 0.02 on the Metrica matches and 0.044 on PFF; the
team features add 0.024 to 0.041 more on the Metrica matches and nothing on PFF and IDSSE, where
the ball is precise or the team easy to read; the decoder's two changes add 0.002 to 0.007 on
Metrica and nothing elsewhere. On plain IDSSE, which loses the ball at random in 41% of play and
is otherwise precise, version 4 equals version 1.

## 2. Assets used and dropped

| Asset | Used? | How, or why not |
|---|---|---|
| Tactical tracking, players and ball | yes | Positions only. The file's ball speed column is filled on 7,377 of 94,214 tracked graded frames and disagrees with the positions, so ball speed is computed from positions |
| Smart Tagging | for one audit, not in the method | 68% of graded frames lie in a phase that names a team. The team implied by the tag agrees with version 1 on 77.5% of those frames when the ball is tracked and on 47.7% when it is not, which is where help would be needed. Half of the frames we label dead lie inside such a phase. No other match has tags, so nothing based on them could be validated. Used once, as a check: the six kick-offs it tags are the six kick-off set-ups our formation rule finds, one to one |
| Homography | no | The tactical tracking is already in pitch coordinates |
| Tactical video | no | We built no detector or tracker of our own |
| Broadcast video and its tracking | no | Not evaluated. That recording is not synchronised to the tactical one and comes without a mapping (DATA.md), and its homography is usable on 65.8% of its frames. Aligning the two would be a project of its own, and no other match has two recordings, so we could not have validated what it added |
| Public 5-minute ground truth | check only | Printed next to every run to catch a broken pipeline. With 17 possessions its score has a 5-95% interval about 0.37 wide, so it cannot choose between versions and was not used to |
| Published full-match scores of the four baselines | for one validation profile, until 2026-11-14 | That profile (`published_baselines`) is fitted to numbers computed from the private ground truth, so since 2026-11-14 it decides nothing and is only reported. Disclosure: before that date it tipped one decision, the rejection of adding our label-free evidence to the model's; measured again later under allowed profiles, that addition stays out |

Tried and dropped, each measured on other matches: estimating stoppage cues from the match itself;
giving the team of the last touch possession until the next touch; a prior that the other team
restarts after a stoppage (it does in 47% of cases); the package's boundary detector inside our
decoder; adding our label-free evidence to the model's; training on 48 PFF matches as well or on
those alone; training the team-feature model on 48 PFF and/or 20 SkillCorner open matches (MIT) as
well, an idea of other teams (their labels change hands later, the model reads possession more
slowly and loses on every Metrica profile; task 17); training labels with short interruptions
merged and a slower decoder (a bet on how the ground truth reads "possession"); a minimum
stoppage length in the decoder, an idea of other teams (it helps only under the still-ball stress
profile; task 16); averaging two or three seeds of the temporal network, with or without the
home/away-swapped view, an idea of other teams (+0.001, below the noise; task 19); a decoder that picks the possessions with the largest expected
`spellF1` under the model's posterior (it never beat the most likely path: the posterior is not
calibrated enough); the temporal network's team head (its 5 Hz output blurs the moment of a change);
weighting the two ball-state models by whether the ball is tracked; moving restarts onto the
ball's re-entry; relabelling short possessions sandwiched by the other team (gains under one
reading of "possession", loses under the other, and on every PFF match); down-weighting evidence
inside ball gaps; shifting possession changes in time; training the team model on labels
debounced 1.5 or 2.5 s, or with extra weight on the frames around a hand-over, ideas of other teams
(task 18: the debounce gains under `controlled` and loses under `touch` and on PFF, the weights
lose everywhere); averaging the team model over the home/away-swapped view (neutral).

## 3. Training and validation data
No label of the provided match was used for training or for choosing anything, and we labelled
nothing by hand on any match.

| dataset | source and licence | labels: who and how | amount | used for |
|---|---|---|---|---|
| IDSSE / Sportec Open DFL Tracking and Event Data: 7 matches (2 Bundesliga, 5 2. Bundesliga, season 2022/23) | Bassek et al., Scientific Data 12:195 (2025), doi:10.1038/s41597-025-04505-y; CC-BY 4.0. Downloaded at run time from https://huggingface.co/datasets/pysport/idsse-data at the fixed revision `a715a38` (2.63 GB), every file checked against `solution/idsse.sha256`; not redistributed | Nobody on our side. The provider's own per-frame flags on corrected tracking (`solution/idsse.py`): `BallStatus` gives D where the ball is out of play, `BallPossession` gives H or A while it is in play | 7 matches, about 1.0 million frames at 25 Hz. Each match is used with home and away swapped as well, in two noise draws | training both submitted models: the temporal network (task 13) and the LightGBM with our team features (task 14). Their tracking is first degraded to resemble the competition tracking; the amounts are calibrated on unlabelled statistics of that tracking. Also validation, strictly out of fold: each match is scored by models trained on the other six |
| Metrica Sports sample data, Sample Games 1-3 | https://github.com/metrica-sports/sample-data (public; no licence file, the README asks for credit; downloaded at run time, not redistributed) | Nobody on our side. The provider's corrected tracking and event files, turned into per-frame H/A/D by `src/possession/validation.py`: D where the tracking has no ball position and the events confirm a stoppage, otherwise the team in possession by the events, under two readings (`touch`, `controlled`) that are both scored | 3 matches, 427,071 graded frames; 1,059 possessions under `touch` (mean 10.2 s), 732 under `controlled` (mean 14.8 s); 32-40% dead | validation and the choice of settings only; nothing is trained on it |
| PFF FC FIFA World Cup 2022: 64 matches | PFF FC (Gradient Sports). Free, on request through the provider's form (https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset); not redistributed and not downloaded by our code | The provider's analysts: an event stream merged into the tracking frames. `solution/pff.py` carries the owner forward between events and marks the ball dead after an out or an end event | 64 matches of broadcast tracking at 29.97 Hz, resampled to 25 Hz. Possession changes hands 4.7 times per live minute, mean 13.1 s | validation only: 16 matches, fixed before measuring, as a test set whose tracking noise is real. The submitted models have seen none of the 64 |

**How validation works.** Three sets of other matches; docs/FUSION.md sections 8-12 give their
roles and the rules.

- **Metrica sample games.** Same provider as the competition tracking. Their tracking is clean, so
  before scoring it is degraded to resemble ours, with three random seeds, under three profiles,
  each fitted to statistics of the competition tracking that need no label:
  - `label_free`: set by hand from aggregates of our tracking. Too easy (the organisers' baseline
    scores 0.334 on it against 0.2433 on the real match).
  - `match_like`: fitted so that our two pipelines agree with each other on the degraded matches as
    they do on the competition match (possessions of one scored against the other: 0.456 on the
    match). As hard as the match, but by losing the ball in 22-26% of play instead of 8%.
  - `match_regime` (task 14): fitted to the match's ball regime, read through version 1's labels:
    the ball lost on 8% of play and 75% of stoppage time, in view during a quarter of the stoppage
    time and lying still before a third of the restarts, 1.9 m from the nearest player. It
    reproduces those, not the match's hardness (its two-pipeline agreement is 0.56).
  - `match_regime_still` (task 15): `match_regime` plus a tracked ball lying still while play goes on
    (a frozen track or a spare ball by a touch line), at the rate that reproduces the match's
    still-ball statistics from its tracking alone (15.5 runs per 10 minutes, twice `match_regime`'s).
    A stress test: it lays the ball still in play about four times more often than the match shows,
    because it also counts the still balls before the match's restarts as play.
- **Weighted by the match's composition** (task 15). The match has more stoppages than the
  validation matches. Version 4 counts stoppages within about 0.5 per 10 minutes of the truth on
  every validation set; on the match it finds 12.1 per 10 graded minutes, against 9.6-10.7 in the
  validation truths, and the excess has the lengths of ordinary stoppages. Every 10-minute validation
  window can be weighted by how often the match has windows of its stoppage density
  (`grid14.py compare --weighted --bins truth:c14_best`). Metrica and IDSSE cover 80% of the match's
  windows, PFF all of them.
- **IDSSE, out of fold**, as degraded for training (the ball lost at random) and in the automatic
  regime. Another provider, and the only labelled data with the ball in view during stoppages.
- **PFF test matches.** Real broadcast tracking, used as delivered: it loses the ball in play like
  the match (6% against 8%) but almost never shows it during stoppages, and its players are mostly
  estimated.
- **`published_baselines`: reported only** (section 2).

## 4. Possession rules implemented

**Two readings of "possession", both always scored.** The other matches' labels come from events,
and the definition can be read two ways: any touch changes the owner (`touch`, about 6.0
possessions per minute of play), or a touch counts only if the team then keeps the ball for 0.2 s
or completes a pass, shot or set piece (`controlled`, about 4.05). We could not tell which the
private ground truth follows; version 4 gains under both.

- **Change of possession.** The team changes only when the evidence for the other team outweighs
  the price of a change. A single touch, a deflection or a header straight back does not carry
  enough; a sequence of frames with the other team in control does. The change is placed on the
  first frame that carries evidence for the new team.
- **Loose and contested balls, and play while the ball is not tracked.** Frames without evidence
  for either team cost both the same, so possession stays with the team that had it, as
  CHALLENGE.md asks.
- **Dead ball.** A stoppage needs evidence that outweighs stopping and restarting (28 in total), so
  short dips in the evidence do not create one. A stoppage that begins within 3 s after the ball
  was lost is dated from the moment it was lost; one that begins within 6 s after the tracked ball
  left the pitch is dated from the moment it left; a restart is dated from the moment the ball
  reappears if it is then moving at 3 m/s or more.
- **Restarts.** The team after a stoppage is whatever the evidence says at the restart; no rule
  ties it to the team before. Measured on other matches, it is the other team in 47% of cases.
- **Kick-offs and the start of the second half.** Dead until the kick, by the formation rule of
  section 1.
- **Abstention.** We never write `X` inside a graded half. `X` is written outside the halves
  only, where nothing is scored.

## 5. Results

**The submitted configuration on matches its models were not trained on** (`spellF1`):

| | Metrica, easier noise | Metrica, noise as hard as the match | Metrica, the match's ball regime | IDSSE, automatic ball | IDSSE, out of fold | PFF, real noise |
|---|---|---|---|---|---|---|
| mean | 0.711 | 0.590 | 0.703 | 0.689 | 0.671 | 0.820 |
| by reading | 0.689 / 0.734 | 0.582 / 0.599 | 0.689 / 0.718 | 0.719 / 0.658 | 0.693 / 0.650 | one reading |
| lowest and highest match | 0.701, 0.721 (3) | 0.577, 0.609 (3) | 0.696, 0.710 (3) | 0.658, 0.727 (7) | 0.642, 0.717 (7) | 0.756, 0.889 (16) |
| difference to version 1 | +0.052 [+0.041, +0.059] | +0.066 [+0.054, +0.074] | +0.049 [+0.040, +0.055] | +0.023 [+0.017, +0.031] | +0.001 [-0.007, +0.005] | +0.046 [+0.039, +0.055] |
| matches won against version 1 | 3 of 3 | 3 of 3 | 3 of 3 | 7 of 7 | 3 of 7 | 16 of 16 |
| team accuracy where both say play | 0.933 | 0.897 | 0.927 | 0.910 | 0.902 | 0.973 |
| play / stoppage F1 | 0.954 | 0.950 | 0.959 | 0.977 | 0.968 | 0.960 |

Readings: Metrica `touch` / `controlled`; IDSSE `controlled1.5` / `controlled3.5`. Intervals are
5-95% over (match, 10-minute window) blocks. Runs: `experiments/runs/20261115T*_c14_best.json`.

**The choice.** Each version is compared with the one before by rules written down before the
measurement that decides (docs/FUSION.md sections 8, 10 and 12; tasks/14-own-directions.md): the
candidate is chosen on two Metrica matches and scored on the third, and the other sets can veto it.
Over the seven configurations measured last, version 4 is chosen on every held-out match.

- Under the team's earlier rule (A), its evidence passes (+0.052 and +0.066, every reading at
  least +0.046, all matches won) and so does PFF; it fails one condition only, winning at least
  five of the seven plain IDSSE matches, where its mean difference is +0.0006 (3 of 7 won).
- Under the rule written for task 14 (B), which replaces win counts on sets that do not resemble
  the match by a bootstrap interval and a floor on losses, it is accepted: +0.059 held out, 5%
  bound +0.050.
- Plain IDSSE loses the ball at random in 41% of play (the match: 8%) and orders the ball-state
  versions unlike every other set; neither of version 4's two main changes can act there. We
  therefore submit version 4, knowing rule A's verdict.

**Checked again against what counts on this match** (task 15, docs/FUSION.md section 13). Weighted
by the match's composition, version 4's gain over version 1 is:
- Metrica: +0.056, +0.068 and +0.050 on the three profiles;
- PFF: +0.041;
- IDSSE in the automatic regime: +0.026.

Unweighted, the same gains are +0.052, +0.066, +0.049, +0.046 and +0.023. Under the still-ball
stress profile (`match_regime_still`), version 1 drops from 0.655 to 0.609 and version 4 leads by
+0.056 [+0.046, +0.062], winning 3 of 3.

Its decoder settings were re-checked by a rule written before measuring. The rule uses weighted
evidence, reads the labels 2/3 `controlled` as the owner fixed, and has loss floors instead of win
counts:
- lower stoppage costs lose;
- a higher price for a change of team (6) gains on the Metrica evidence (+0.010 held out) but loses
  on the PFF matches (-0.007 weighted to the match, -0.004 unweighted), the only set with real
  noise.

So version 4 is unchanged. That configuration is kept as version 5 (`configs/c15_best.toml`, sha256
`0167a8c6...`, reproduced like version 4). It is better than version 4 on six of seven sets and is
the only one of the two that also passes the team's earlier rule A: plain IDSSE won 6 of 7 against
version 1. It is a bet on the `controlled` reading (+0.012 to +0.020 under it, -0.010 to +0.005
under `touch`).

**Ideas of other teams, tested against version 4 afterwards** (tasks 16-19, 2026-11-15; docs/FUSION.md
sections 14-16). Each family was measured under the same rule, written before measuring, and each
idea is credited to the teams it came from in docs/DECISIONS.md. None was adopted:

| Idea (task) | Best member, held out | Why it stays out |
|---|---|---|
| Minimum stoppage length in the decoder (16) | +0.015, with a change price of 6 | PFF -0.008 against a floor of -0.005; on version 4's decoder alone +0.005, all from the still-ball stress profile |
| Team model also trained on 48 PFF and/or 20 SkillCorner matches (17) | -0.008 | Loses on every Metrica profile, mostly under `touch` |
| Team training for `spellF1`: labels debounced, hand-overs weighted, swapped view (18) | +0.016 | `touch` readings down to -0.041 and PFF -0.087 |
| Two or three seeds of the temporal network, swapped view (19) | +0.001 | Below the threshold (+0.005); safe on every floor |

Every gain that survives the held-out choice comes from reading possession as changing hands more
slowly, and PFF, the only set with real noise, refuses each of them. All options stay in the code,
off by default; version 4's file is unchanged.

**Beside two other teams, on the same conditions** (`scripts/compare_teams.py`; their predictions
come from their public repositories, and the truths are checked to be identical to ours). Both teams
trained on Metrica and IDSSE, so PFF is the only set none of the three of us trained on. There,
version 4 scores 0.820 against 0.743 (Barca-Hackathon) and 0.767 (Team Sigue), and is higher on
all 16 matches against each. On Metrica, which is their training data, they score 0.65-0.75
against our 0.59-0.71.

**Where the rest is.** Replacing one input by the truth on the Metrica matches (version 1, the
harsher published profile): perfect stoppages gave +0.07, a perfect team on frames without a ball
+0.08, a perfect team on frames with a ball +0.23. Version 4 attacks the first and the last;
possessions shorter than 5 s and the first seconds after a change of possession remain the main
loss.

**On the competition match, without labels:** 271 possessions (4.91 per live minute, mean 12.2 s),
49% of graded frames dead (version 1: 300 possessions, 5.59 per minute). 7.6% of graded frames are
labelled differently from version 1. Run: `experiments/runs/20261115T004905_mikolajekiert_c14_best.json`.

**Public 5 minutes, as a check only:** `spellF1` 0.723, 14 possessions predicted against 17 true.
It confirms the pipeline is aligned and decides nothing (section 2).

**What we expect.** Nearer the figures of the two profiles fitted to the match (0.59-0.70) than
PFF's. These figures rank our own versions; they are not a forecast: the noise is simulated where
it is not another provider's, the reading of the labels is unknown, and part of what makes the
match hard (section 6) is simulated by none of the sets.

## 6. Limitations

- **We do not know which reading of possession the ground truth follows.** The price for a change
  of team (5) is the value that does not lose under either reading; version 4 gains under both. A
  price of 6 (version 5) would gain about 0.01 more if the ground truth reads `controlled`.
- **The noise is simulated on the matches that share our provider.** The match's ball regime is
  reproduced (`match_regime`), its hardness only through a larger ball offset (`match_like`): our
  two pipelines agree on 0.46 of the match's possessions, and no simulated mechanism gets there
  with the ball as close to the players as on the match. A ball lying still in the wrong place is
  simulated since task 15, more often than on the match; identity switches and duplicated players
  are not simulated anywhere.
- **A fifth of the match is outside the validation.** By version 4's labels, 20% of its 10-minute
  windows have 15 or more stoppages, a density no Metrica or IDSSE window reaches (PFF has a few).
- **The team features help where the ball is off its possessor.** They gain on all three Metrica
  profiles and change nothing on IDSSE and PFF, whose ball is precise or whose team is easy to read.
  How much of the match's difficulty is of that kind is not known.
- **Few validation matches of our provider** (three), used for many comparisons (more than 160
  configurations): the choice on two matches and scoring on the third corrects for the selection
  within a family, not across families. That is why the other sets are kept beside them.
- **Both models are trained on seven matches** of another league and another tracking system.
- **We did not use the video.**

## 7. Reproduction

```bash
uv sync                                   # Python 3.11, every dependency pinned by uv.lock
# unpack the data bundle into ./data as DATA.md describes (or set MTAH_DATA_DIR)
make submit CONFIG=configs/c14_best.toml TEAM=culers
# without make: uv run possession submit configs/c14_best.toml --team culers
```

This writes `submission/culers.csv`, runs the pipeline a second time from a fresh parse of the
data, requires the same sha256, calls the organisers' validator and prints the sha256 above. It
takes under a minute.

- **Both trained models are in the repository**, so nothing is trained or downloaded:
  the LightGBM with our team features in `models/team/` (2.8 MB, with the sha256 of its text) and
  the temporal network in `models/sequence/` (1.8 MB, with its sha256). Each is used only if it is
  exactly the model the current code would train (the file name is the training code's cache key).
  With the team model removed from the cache, the pipeline unpacks it and writes the same file.
- **To retrain the LightGBM from scratch instead:** `MTAH_NO_CACHE=1 uv run possession run
  configs/c14_best.toml` (2.63 GB of open data, a few minutes). With 8 threads, LightGBM's default on
  the machine we used, the retrained model is byte-identical to the committed one; with 4 threads it
  is not (its histogram sums depend on the number of threads), which is why the model is committed.
  The temporal network is retrained with `uv run --with torch==2.15.0 possession train-sequence
  --folds --final` (about 1.5 hours on a CPU); its numpy inference reproduces torch to 1e-14.
- **Checked on:** macOS arm64, and from a fresh clone with an empty cache in a Linux arm64 container
  and an emulated Linux x86-64 container (Debian 12; `scripts/repro_container.sh`): the same file,
  the organisers' validator passing, each time. Not checked: Windows, a real x86-64 processor.
- **Not the submission path:** `python -m solution.submission` and the notebooks in `notebooks/`
  are the supervised track's own experiments. They use another decoder and produce another file.
