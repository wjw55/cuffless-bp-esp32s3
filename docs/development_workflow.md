# Development and branch workflow

## Branches isolate changes, not categories of files

- `main` is the known working software baseline for collection and demonstrations.
  This label does not establish clinical validation or hardware qualification.
- `rtos-trial` preserves the separate RTOS acquisition experiment. Do not merge it
  merely to simplify the branch list; compare acquisition and fault behaviour first.
- Create one short-lived branch per change, such as `chore/tooling-cleanup`,
  `experiment/ppg-denoising` or `feature/dashboard-recording`. These are naming
  examples, not authorization to implement those features.

There is no permanent `testing` branch used as a file dump. Every branch is a
complete project snapshot. Switching branches does not normally remove ignored
recordings, model packages or build directories.

## Normal workflow

Start from a clean, up-to-date `main`, then create a descriptive branch:

```powershell
git status --short --branch
git switch main
git switch -c feature/your-change
```

Keep the change focused. Preserve raw formats, model/config compatibility and
existing commands unless the task explicitly requires a migration. Run the full
regression suite before committing:

```powershell
& 'C:\wjw\Anaconda\python.exe' -m unittest discover -s tools\tests -p 'test_*.py'
git diff --check
```

Inspect the exact staged files and commit only software, tests and documentation.
Keep participant information and generated models/results out of Git. Review the
branch and perform relevant hardware tests before merging it into `main`. A Python
test pass alone does not qualify firmware, BLE range, power or physiological accuracy.

After approval, a branch based directly on `main` can be integrated without a
merge commit:

```powershell
git switch main
git merge --ff-only feature/your-change
```

If the fast-forward fails, stop and inspect the divergence. Do not force-reset,
overwrite local work or delete the experiment to make it fit. Keep important
experimental results reproducible even when the implementation is not deployed.

## Files and editor clutter

| Location | Handling |
| --- | --- |
| `main/`, `tools/`, `config/` | Tracked source and configuration |
| `tools/experiments/` | Archived, runnable research with retained tests |
| `docs/archive/` | Historical measurement reports |
| `data/raw/`, `data/labels/`, `data/quarantine/` | Local participant information; ignored and securely backed up |
| `data/processed/`, `data/models/`, `data/external/` | Local results, model packages and external datasets; ignored |
| `hardware/`, `releases/`, `build/`, `build_*/` | Local hardware/build artifacts; ignored |
| `__pycache__/`, `.pytest_cache/`, `.ruff_cache/` | Regenerable caches; ignored |

Open `ppg_logger.code-workspace` in VS Code for an optional view that hides
generated build/cache folders. The existing ignored `.vscode/settings.json` also
has the same Explorer exclusions for this checkout. Nothing is deleted, and
recordings/model directories remain visible. These settings do not change the
Codex app's file browser or affect firmware builds and Python imports.

Supported fallback viewers and shared modules remain in place. In particular,
`view_live_bp.py` supplies BP state/inference/display rules used by the desktop
and dashboard, and `view_live_upper_arm_hr.py` is used by collector validation.
Do not archive or delete them based only on their filenames.
