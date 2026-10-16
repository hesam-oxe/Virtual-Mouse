# Security Policy

## Supported Versions
Only the `main` branch is supported with security updates.

## Reporting a Vulnerability
Please use GitHub Private Vulnerability Reporting:
`https://github.com/hesam-oxe/Virtual-Mouse/security/advisories/new`

Do NOT open a public issue for security bugs. You will receive a response within 72 hours.

## Rules
- Never commit tokens (`ghp_*`), `.git-credentials`, `.env`, or `*.pem`.
- Model downloads must be HTTPS with SHA256 verification (see `GOD_PLAN.md` P0).
- Webcam frames are processed locally and never uploaded.
- Dependencies are pinned in `requirements.txt`; Dependabot opens PRs weekly.
