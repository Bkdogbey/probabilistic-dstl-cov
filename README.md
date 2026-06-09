# probabilistic-dstl-cov



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📁 Project Structure

```{=bash}
├── src/
│   ├── data/           # Data loading and saving scripts
│   ├── dataset/        # Dataset preparation and loaders
│   ├── features/       # Feature engineering
│   ├── models/         # Model training and evaluation
│   ├── visualization/  # Plots and visualizations
│   └── **init**.py
│
├── configs/
│   └── config.yml      # YAML configuration file
│
├── main.py             # Entry point
├── utils.py            # Utility functions including skip_run context manager
├── requirements.txt    # Project dependencies
├── pyproject.toml      # Project and tool configuration
├── .pre-commit-config.yaml
├── README.md
└── tests/              # Unit and integration tests

```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/iHuman-Lab/probabilistic-dstl-cov.git
cd probabilistic-dstl-cov
```

### 2. Always use a conda virtual environment

```bash
   conda activate myenv
```

   *(Replace `myenv` and Python version as needed)*

---

## 🛠️ Pre-commit Setup

This project uses [pre-commit](https://pre-commit.com/) to automatically check your code before commits and pushes, ensuring consistent code quality and style.

### How to install and enable pre-commit hooks:

1. **Ayour conda environment**

   ```bash
   conda activate myenv
   ```

   *(Replace `myenv` and Python version as needed)*

2. **Install `pre-commit` in your conda environment**

   ```bash
   conda install -c conda-forge pre-commit
   ```

   or if you prefer:

   ```bash
   pip install pre-commit
   ```

3. **Run pre-commit checks manually (optional)**

   To manually run all pre-commit checks on all files:

   ```bash
   pre-commit run --all-files
   ```

---

### Notes:

* Always activate your conda environment before running Git commands to ensure hooks work properly.
* If you update `.pre-commit-config.yaml`, run `pre-commit install` again to update the hooks.
* The `pre-push` hook will run all pre-commit checks before pushing. If any check fails, the push will be aborted.



---

## ⚙️ Configuration

The project uses a YAML config file at `configs/config.yml`.

```python
import yaml

with open("configs/config.yml") as f:
    config = yaml.safe_load(f)
```

Modify this file to adjust parameters, paths, and flags.

---

## ⏯️ Conditional Execution with `skip_run`

Use the `skip_run` context manager from `utils.py` to control which pipeline blocks run.

Example from `main.py`:

```python
from utils import skip_run

with skip_run("skip", "Data") as check, check():
    # This block will be skipped if flag is "skip"
    ...
```

* Pass `"skip"` to skip blocks
* Pass `"run"` (or anything else) to execute blocks

The utility prints colored console messages indicating running/skipping status.

---

## 🧹 Code Quality

This project uses pre-commit hooks for automated linting and formatting:

* `ruff` for linting and enforcing naming conventions
* `codespell` for typo detection

### Install pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```


---

## 📝 Running the Project

Start the main pipeline:

```bash
python main.py
```

---

## 📜 License

MIT License © Your name (or your organization/company/team)

---

## 🙏 Acknowledgements

Project scaffolded using a custom Cookiecutter template for Python data science projects.

