# SweetBITS

Bioinformatics command-line tools for the SweBITS project.

See `GEMINI.md` for detailed specifications and architecture.

## Installation

```bash
# Optional: install JolTax in editable mode
pip install -e /home/daniel/devel/JolTax

# Install SweetBITS in editable mode
pip install -e /home/daniel/devel/SweetBITS
```

## Shell Autocompletion

SweetBITS supports shell autocompletion. To enable it for Bash, add this to your `~/.bashrc`:

```bash
eval "$(_SWEETBITS_COMPLETE=bash_source sweetbits)"
```

*Note: For Zsh, simply replace `bash_source` with `zsh_source` in the command above and add it to your `~/.zshrc`.*

### For Conda Users
If you use Conda, the command above might cause "command not found" errors when opening a new terminal if the environment containing SweetBITS is not active. To ensure autocompletion only loads when the environment is actually active, use a Conda activation script:

```bash
# 1. Activate your environment
conda activate your_env_name

# 2. Create the activation directory
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# 3. Add the completion command
echo 'eval "$(_SWEETBITS_COMPLETE=bash_source sweetbits)"' > $CONDA_PREFIX/etc/conda/activate.d/sweetbits_completion.sh
```
