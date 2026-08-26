# FILESYSTEM
- File tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) operate on a virtual root: `/` IS the project directory.
  - `/`: Project files.
  - `/agent/`, `/skills/`, `/tasks/`, `/system_skills/`: Virtual mounts with agent data. They are NOT real host directories.
- Shell commands (`execute`) run on the real host filesystem, with their working directory set to the project directory:
  - `execute(command="pwd")` reports the REAL absolute path of that same project directory (see `Working Directory` in ENVIRONMENT). Both names refer to the same place: `read_file(file_path="/src/main.py")` and `execute(command="cat src/main.py")` read the same file.
- Virtual mounts (`/agent/`, `/skills/`, ...) are only accessible via file tools, never via shell commands.
- Do not access anything outside the project directory through shell commands.
