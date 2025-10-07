# Agent Operating Checklist

## Before Work Session
- [ ] Review last diary entry and outstanding TODO items.
- [ ] Sync local repo state and note uncommitted changes.
- [ ] Confirm target checklist items for the session.
- [ ] Verify data availability and config files referenced today.

## During Work Session
- [ ] Update plan tool with current step before major actions.
- [ ] Keep logs clean; no direct `print`, rely on `logging`.
- [ ] Validate assumptions with small probes (notebooks or scripts).
- [ ] Record key decisions, parameter choices, and blockers in diary.

## After Work Session
- [ ] Run smoke tests or validation scripts relevant to touched modules.
- [ ] Execute debug two-epoch training run (`python -m scripts.train --config configs/debug_two_epoch.yaml --run-name <alias>`) to ensure end-to-end behaviour remains stable.
- [ ] Capture metrics snapshots or plots produced.
- [ ] Update project checklist item statuses.
- [ ] Write a diary summary with outcomes and next steps.
- [ ] Stage commits or stash work-in-progress as appropriate.
