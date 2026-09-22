# Code Style

## Environment

- Use the `.venv` directory at the repository root for all Python commands.
- Run tools through `.venv/bin/python`, `.venv/bin/pytest`, or the corresponding executable in `.venv/bin`.

- Prefer simple, direct implementations with a clear main flow.
- Define helper functions only when they provide real reuse or substantially improve readability. Avoid thin wrappers and deep call chains.
- Keep functions focused and reasonably short, but do not split them into many trivial helpers.
- Minimize branching. Prefer clear data flow and early returns over nested conditions or long `if`/`elif` chains.
- Do not add defensive validation or exception handling by default. Check inputs only at necessary external boundaries or when explicitly required.
- Avoid handling hypothetical edge cases that are outside the stated requirements.

- Coding style should be simple, easy to read.