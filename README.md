# hcr_data_qc

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-87.8%25-yellow)
![Coverage](https://img.shields.io/badge/coverage-6%25-red?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

## About

Quality control analysis for AIND HCR data processing. Provides tools for validating tile alignment, camera alignment, segmentation, spectral unmixing, and spot detection.

**Use Cases:**
1. **Interactive analysis** - Within a CodeOcean capsule/cloud workstation (or local machine):

    + Make sure processed HCR data asset is attached to capsule


   ```bash
   // run just tile_alignment
   python launch_qc.py --dataset HCR_788639-25_2025-06-06_13-00-00_processed_2025-06-17_07-08-14 --output-dir /root/capsule/scratch/qc-test --tile-alignment --pyramid-level 4
   ```

   ```bash
   // run all qc 
   python launch_qc.py --dataset HCR_788639-25_2025-06-06_13-00-00_processed_2025-06-17_07-08-14 --output-dir /root/capsule/scratch/qc-test --all --pyramid-level 0 
   ```
2. **Reproducible runs** - With CodeOcean app panel
    + See [HCR QC Kickoff capsule](https://codeocean.allenneuraldynamics.org/capsule/8714887/tree)
4. **Pipeline integration** - As automated QC steps *(not implemented yet)*

Intergration will AIND QC portal will happen when team identifies and evaluates essential plots.


## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
pip install -e .[dev]
```

## Contributing

### Linters and testing
Run `pre_commit_checks.py`, which includes coverage, black, isort, flake8, & interrogate

### Pull requests

+ **Internal members** please create a branch. 
+ **External members** please fork the repository and open a pull request from the fork. 

### Commits
We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```text
<type>: <short summary>
```

type is one of:

- **build**: Changes that affect build tools or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bugfix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests
