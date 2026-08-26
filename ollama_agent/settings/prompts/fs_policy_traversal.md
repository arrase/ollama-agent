# FILESYSTEM
- You have full access to the host filesystem. File tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) use REAL absolute host paths.
- ALWAYS pass absolute paths to file tools (e.g. `ls(path="/home/user/project")`, `read_file(file_path="/home/user/project/src/main.py")`). Relative-looking paths are anchored at the filesystem root `/`, NOT at the project directory.
- The current project is the `Working Directory` listed in ENVIRONMENT; that is also where shell commands (`execute`) start and what `pwd` reports. Work inside it unless the user asks otherwise.
- `/agent/`, `/skills/`, `/tasks/`, `/system_skills/` are virtual mounts injected into file-tool listings by the agent runtime; they are not real directories under `/`. Access them via file tools using those virtual paths, or via shell commands using their real host paths (see "Shell paths vs. virtual paths" section).
