import os
import importlib.metadata

script_dir = os.path.dirname(os.path.abspath(__file__))
req_path = os.path.join(script_dir, "requirements.txt")

# Get installed packages (Python 3.8+)
installed = {
    dist.metadata["Name"].lower(): dist.version
    for dist in importlib.metadata.distributions()
}

updated_lines = []

with open(req_path, "r") as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith("#"):
            updated_lines.append(line)
            continue

        pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip().lower()

        if pkg_name in installed:
            updated_lines.append(f"{pkg_name}=={installed[pkg_name]}")
        else:
            updated_lines.append(line)

with open(req_path, "w") as f:
    f.write("\n".join(updated_lines))

print("requirements.txt updated to current installed versions.")