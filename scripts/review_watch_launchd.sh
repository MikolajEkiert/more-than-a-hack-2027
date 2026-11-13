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
    mkdir -p "$LOG_DIR" "$(dirname "$PLIST")"
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string><string>$REPO_DIR/scripts/pr_review_watch.sh</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO_DIR</string>
  <key>StartInterval</key><integer>${2:-300}</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$LOG_DIR/review.log</string>
  <key>StandardErrorPath</key><string>$LOG_DIR/review.log</string>
</dict></plist>
PL
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "installed: every ${2:-300}s, log $LOG_DIR/review.log" ;;
  uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed" ;;
  status)
    launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E "state|last exit" || echo "not installed"
    tail -n 5 "$LOG_DIR/review.log" 2>/dev/null || true ;;
  *) echo "usage: $0 install [seconds] | uninstall | status"; exit 1 ;;
esac
