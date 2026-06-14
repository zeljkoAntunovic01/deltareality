# Computer Vision Assignment

## 1. Environment Setup

This project uses a lightweight Python environment based on:

- Python 3.12.8
- `venv`
- `pip`
- `requirements.txt`


### Create the virtual environment

From the repository root:

```bash
python3.12.8 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Upgrade the base packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

### Adding new dependencies

When a new dependency is needed, add it first to `requirements.in`.

For example:

```txt
numpy
open3d
opencv-python
```

Then install or update the environment:

```bash
pip install -r requirements.in
```

After verifying that the dependency is needed and works correctly, freeze the exact installed versions:

```bash
pip freeze > requirements.txt
```

Commit both files:

```bash
git add requirements.in requirements.txt
git commit -m "Add project dependencies"
```

This keeps a simple distinction between:

- `requirements.in`: direct dependencies chosen by the developer
- `requirements.txt`: exact pinned dependency versions used to reproduce the environment

### Recreating the environment from scratch

A reviewer can recreate the environment with:

```bash
git clone <repository-url>
cd <repository-name>

python3.12.8 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
git clone <repository-url>
cd <repository-name>

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 2. Assignment Workflow

This section documents the steps taken during development.

### Step 1: Repository and environment setup

Created the initial repository structure with:

- `.venv` for the local Python virtual environment
- empty `requirements.in`
- empty `requirements.txt`
- empty `requirements-dev.in`
- empty `requirements-dev.txt`
- `.gitignore`
- `README.md`

The virtual environment itself is not committed to Git. It is excluded through `.gitignore`.

### Step 2: Assignment analysis

_To be filled in after receiving and reviewing the assignment prompt._

### Step 3: Implementation approach

_To be filled in during development._

### Step 4: Results and assumptions

_To be filled in during development._

### Step 5: Resources used

_To be filled in during development._