#!/bin/zsh
cd "$(dirname "$0")/.." || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print 'Python environment missing. Follow README.md setup first.'
  read '?Press Enter to close.'
  exit 1
fi
print 'Adjust your position in PREVIEW, then press SPACE to record 25 seconds.'
if ! .venv/bin/python assignment/assignment_starter.py --record --duration 25 "$@"; then
  print '\nCounter could not finish. See the error above.'
fi
read '?Press Enter to close.'
