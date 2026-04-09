# Release Checklist

Use this checklist before creating a release tag like `v1.0.0`.

## 1) Code + CI Health

- [ ] `main` is green on latest commit.
- [ ] CI workflow passed on both Linux and Windows.
- [ ] No failing or flaky tests in the last 24h.
- [ ] No unresolved high-severity issues/PR comments.

## 2) Training/Runtime Validation

- [ ] `python test_models.py` passes.
- [ ] `python production_smoke.py` passes.
- [ ] Text generation starts and shows progress on supported device settings.
- [ ] Tabular and image training each complete at least a 1-epoch sanity run.

## 3) Version + Notes

- [ ] Version/tag chosen (semantic versioning): `vMAJOR.MINOR.PATCH`.
- [ ] Changelog/release notes include:
  - [ ] user-facing changes
  - [ ] breaking changes
  - [ ] migration notes (if any)
  - [ ] known limitations

## 4) Security + Compliance

- [ ] No secrets/credentials in diff.
- [ ] Dependency changes reviewed.
- [ ] License-sensitive changes reviewed.

## 5) Create Release

- [ ] Pull latest `main`.
- [ ] Create annotated tag:
  - `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Push tag:
  - `git push origin vX.Y.Z`
- [ ] Confirm `Release` workflow completed and artifacts uploaded.

## 6) Post-Release

- [ ] Smoke-test release artifact in clean environment.
- [ ] Announce release (summary + upgrade notes).
- [ ] Open follow-up issues for deferred items.
