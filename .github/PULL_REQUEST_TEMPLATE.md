## Summary of Changes

A concise description of what this PR accomplishes and why.

## Issue Reference
Fixes #(issue number)

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Security fix / hardening
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing Checklist
- [ ] Added new unit/integration tests for the changes
- [ ] Ran `pytest tests/test_regression.py tests/test_api_security.py` (all tests passing)
- [ ] Verified no plaintext secrets or credentials committed
- [ ] Validated with `python argus.py doctor`

## Privacy & Security Verification
- [ ] No raw biometric embeddings or PII written unencrypted to storage
- [ ] Mutating routes are protected by appropriate RBAC role dependencies
- [ ] Audit trail records privileged mutations
