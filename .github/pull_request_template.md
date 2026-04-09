## What changed

- 

## Why this change

- 

## Validation

- [ ] `python -m compileall -q core training data ui start.py test_models.py production_smoke.py`
- [ ] `python test_models.py`
- [ ] `python production_smoke.py`
- [ ] Manual check completed (if UI/runtime behavior changed)

## Risk assessment

- [ ] Low risk
- [ ] Medium risk
- [ ] High risk (explain below)

Risk notes:

- 

## Release impact

- [ ] No user-facing impact
- [ ] User-facing behavior changed
- [ ] Breaking change

If user-facing or breaking, include upgrade/migration notes:

- 

## Checklist

- [ ] CI is green (Lint + Linux + Windows jobs)
- [ ] No secrets or credentials in diff
- [ ] Docs updated (if behavior/config changed)
- [ ] Reviewer can reproduce test plan
