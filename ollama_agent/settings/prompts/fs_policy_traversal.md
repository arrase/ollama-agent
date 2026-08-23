# FILESYSTEM
- You have access to the entire host filesystem.
- For file tools (`read_file`, `write_file`, `ls`, `grep`), `/` is the host system root (`/etc`, `/home`, etc.).
  - To access project files, use relative paths (e.g., `read_file(file_path="src/main.py")`) or inspect the project with `ls(path=".")`.
  - To discover the absolute project path, run `execute(command="pwd")` (Unix/macOS) or `execute(command="cd")` (Windows).
- `/agent/` (memory) and `/skills/` are special virtual folders accessible only via file tools, not via shell commands (`execute`).
