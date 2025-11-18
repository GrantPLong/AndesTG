# Virtual Environment Setup

**IMPORTANT**: The `venv/` directory is NOT included in the git repository (it's in .gitignore).

## Creating the Virtual Environment

Each user needs to create their own virtual environment:

```bash
cd python
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Why Virtual Environments Are Not Committed

Virtual environments contain:
- Platform-specific binaries
- Absolute file paths
- Compiled Python bytecode
- Library installations specific to your system

These cannot be shared across machines/users. Each person must create their own.

## Activating the Environment

```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Your prompt will show `(venv)` when active.

## Deactivating

```bash
deactivate
```

## If You Get Import Errors

If you see "No module named..." errors:

1. Make sure venv is activated: `source venv/bin/activate`
2. Reinstall dependencies: `pip install -r requirements.txt`
3. Or recreate the venv: `rm -rf venv && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
