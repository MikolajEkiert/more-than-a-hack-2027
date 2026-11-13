# <Team name>: frame-level ball possession

> Deliverable 3 (SUBMISSION.md). Keep it to a few pages. Maintained with `/report`, from
> docs/DECISIONS.md, the task logs and experiments/runs/.

## 1. Approach
_What we built, and why this rather than something else._

## 2. Assets used and dropped
| asset | used? | how / why not |
|---|---|---|
| tactical ATD (players + ball) | | |
| Smart Tagging | | |
| homography | | |
| tactical video | | |
| broadcast video + ATD | | |

## 3. Training and validation data
_Every labelled dataset: source, who labelled it, how, how much. None from the provided game.
The public 5-minute window was used only to check the pipeline is correct._

## 4. Possession rules implemented
_Loose balls, challenges, deflections, restarts, dead-ball detection, when we abstain (X)._

## 5. Results
_Label-free diagnostics and other-game validation, each number citing its experiments/runs file.
Public-window numbers, if shown, are labelled as a sanity check._

## 6. Limitations
_Where it fails, and what we would fix with more time._

## 7. Reproduction
```bash
uv sync                                   # Python 3.11, pinned by uv.lock
# unpack the data bundle into ./data (or set MTAH_DATA_DIR)
make submit CONFIG=configs/<final>.toml TEAM=<team>
```
config: `configs/<final>.toml` · commit: `<sha>` · sha256 of `<team>.csv`: `<sha256>`
