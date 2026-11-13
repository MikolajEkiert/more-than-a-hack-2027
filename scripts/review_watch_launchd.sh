#!/usr/bin/env bash
# Install / remove the launchd job that runs scripts/pr_review_watch.sh every few minutes.
#   scripts/review_watch_launchd.sh install [interval_seconds]   (default 300)
#   scripts/review_watch_launchd.sh uninstall
#   scripts/review_watch_launchd.sh status
# Run it on ONE machine only, or every PR gets one review per teammate.
set -euo pipefail
LABEL="com.mtah.pr-review"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$HOME/Library/Logs/mtah-pr-review"

case "${1:-}" in
  install)
    # The job runs from STATE_DIR with its own clone: background jobs may not read ~/Desktop or
    # ~/Documents (macOS TCC), and the review must not depend on the state of a working copy.
    STATE_DIR="$HOME/Library/Application Support/mtah-pr-review"
    mkdir -p "$LOG_DIR" "$(dirname "$PLIST")" "$STATE_DIR/bin"
    if [ ! -d "$STATE_DIR/repo/.git" ]; then
      git clone -q "$(git -C "$REPO_DIR" remote get-url origin)" "$STATE_DIR/repo"
    fi
    cp "$REPO_DIR/scripts/pr_review_watch.sh" "$STATE_DIR/bin/pr_review_watch.sh"
    # Trusted review instructions, from this working copy (never from the PR under review).
    cp "$REPO_DIR/.claude/skills/review-pr/SKILL.md" "$STATE_DIR/bin/review-pr.md"
    cp "$REPO_DIR/AGENTS.md" "$STATE_DIR/bin/AGENTS.md"
    chmod +x "$STATE_DIR/bin/pr_review_watch.sh"
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string><string>$STATE_DIR/bin/pr_review_watch.sh</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>PR_REVIEW_REPO_DIR</key><string>$STATE_DIR/repo</string>
  </dict>
  <key>WorkingDirectory</key><string>$STATE_DIR</string>
  <key>StartInterval</key><integer>${2:-300}</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$LOG_DIR/review.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/review.log</string>
</dict></plist>
PL
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    # bootout is asynchronous (it waits for a running review); bootstrap fails until it is done.
    for _ in $(seq 1 60); do launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || break; sleep 2; done
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "installed: every ${2:-300}s, log $LOG_DIR/review.log"
    echo "(re-run install after changing scripts/pr_review_watch.sh to update the job's copy)" ;;
  uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed (state, clone and log stay in ~/Library/Application Support/mtah-pr-review and $LOG_DIR)" ;;
  status)
    launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E "state|last exit" || echo "not installed"
    tail -n 5 "$LOG_DIR/review.log" 2>/dev/null || true ;;
  *) echo "usage: $0 install [seconds] | uninstall | status"; exit 1 ;;
esac
