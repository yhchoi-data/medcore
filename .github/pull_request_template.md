<!-- Usually update Summary / Why / Changes / Validation / Review Notes for each PR. -->

## Version
- git tag:
- release needed:

## Summary
- improve DICOM reader behavior in `medcore`
- add developer tooling (`ruff`, `pytest`, `pre-commit`)
- update package documentation and installation guidance

## Why
- DICOM series / acquisition handling needed more consistent behavior
- metadata handling for multi-series cases needed to be more reliable
- the package did not yet have a standard lint / test / pre-commit workflow
- README / USAGE needed to reflect the current package structure and developer setup

## Changes
- refine `ImageReader` DICOM loading, metadata handling, and series selection behavior
- improve multi-series / acquisition handling paths
- add `pyproject.toml` for `ruff` and `pytest` configuration
- add `.pre-commit-config.yaml`
- add pinned `requirements.txt`
- add smoke test for public imports
- update `setup.py` with `dev` extras
- update `README.md` and `USAGE.md`

## Validation
- `ruff check .`
- `python -m pytest test/test_imports.py -q`
- `pre-commit run --all-files`

## Impact
- affects `packages/medcore` package behavior and development workflow
- intended to preserve public package usage while improving consistency
- adds standard developer tooling for ongoing maintenance

## Review Notes
- please focus on `ImageReader` DICOM series selection and metadata behavior
- please check whether the current dev tooling scope is appropriate for this package
- please review documentation updates for correctness and clarity

## Checklist
- [ ] lint passed
- [ ] tests passed
- [ ] docs updated
- [ ] backward compatibility checked
