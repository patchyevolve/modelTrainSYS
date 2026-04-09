# Branch Protection Policy

Use this policy for `main` (and any long-lived release branch like `release/*`).

## Recommended GitHub Settings

- **Require a pull request before merging**
  - Require approvals: `1` (minimum)
  - Dismiss stale approvals when new commits are pushed
  - Require review from code owners (enable when `CODEOWNERS` is added)
- **Require status checks to pass before merging**
  - Required checks:
    - `Lint (fast)`
    - `ubuntu-latest / py3.12`
    - `windows-latest / py3.12`
  - Require branches to be up to date before merging
- **Require conversation resolution before merging**
- **Require linear history**
- **Do not allow force pushes**
- **Do not allow deletions**

## Admin/Security Recommendations

- Include administrators in branch protection.
- Restrict who can push directly to protected branches (prefer PR-only).
- Enable secret scanning and push protection at repository level.

## Merge Strategy

- Preferred: **Squash merge** for feature branches.
- Keep commit title descriptive and linked to intent (feature/fix/refactor/test/docs).

## Operational Rule of Thumb

- No direct commits to `main`.
- Every production-impacting change must pass CI and smoke tests.
- Every release tag (`v*`) should be created from a green commit on protected `main`.
