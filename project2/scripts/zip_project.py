"""
Automatically compress all relevant project files into a ZIP archive ready for submission.

Examples
--------
```
python scripts/zip_project.py --config scripts/zip_project.yaml --project 1 --student 12345678
```
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Sequence

import yaml


def deserialize(path: Path) -> Any:
    """Read a configuration file in JSON or YAML format.

    Parameters
    ----------
    path : Path
        Path to the configuration file.

    Returns
    -------
    Any
        The configuration data.

    Raises
    ------
    ValueError
        If the file is not a JSON or YAML file.
    """
    extension = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if extension in {".json"}:
        return json.loads(text)
    if extension in {".yml", ".yaml"}:
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config extension: {extension}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Redact Python function bodies (keep docstrings) and zip outputs.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file.",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project number.",
    )
    parser.add_argument(
        "--student",
        required=True,
        help="Student ID.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    config = deserialize(config_path)
    files = config.get("files", [])
    if not isinstance(files, list) or not files:
        sys.exit("No files to zip.")

    zip = Path(f"aso_project_{args.project}_student_{args.student}.zip")
    zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip, "w", compression=zipfile.ZIP_DEFLATED) as zip_deflated:
        for file_index, file in enumerate(files, 1):
            if isinstance(file, str):
                path_str = file
            else:
                path_str = file.get("path")

            if not path_str:
                print(f"No path for file {file_index}.", file=sys.stderr)
                continue

            path = Path(path_str)
            if not path.exists() or path.is_dir():
                print(f"{path_str} is not a file.", file=sys.stderr)
                continue

            arcname = path.as_posix()
            with path.open("rb") as f:
                zip_deflated.writestr(
                    zinfo_or_arcname=arcname,
                    data=f.read(),
                )

    print(f"Successfully zipped project files: {zip}")


if __name__ == "__main__":
    main()
