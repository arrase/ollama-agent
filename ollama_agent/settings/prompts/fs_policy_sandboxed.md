# FILESYSTEM
- For file tools (`read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`), the root path `/` is the project directory.
  - `/`: Project files.
  - `/agent/`: Internal data and memory (virtual directory).
  - `/skills/`: Available skills (virtual directory).
- For shell commands (`execute`), your working directory is the project folder. Use relative paths (e.g., `python script.py`).
- Virtual directories (`/agent/`, `/skills/`) are only accessible via file tools, not via shell commands.
