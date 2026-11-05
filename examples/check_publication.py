"""Check selected public files for broken local links and obvious private material."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SENSITIVE = re.compile(
    r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|"
    r"\bAKIA[0-9A-Z]{16}\b|\bsk-[A-Za-z0-9_-]{20,}|"
    r"/Users/[^/\s]+/|[A-Z]:\\(?:strata|Users)\\|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)


def main():
    errors = []
    files = [path for path in ROOT.rglob("*") if path.is_file()
             and not any(part in (".git", "__pycache__", ".venv", "demo-output") for part in path.relative_to(ROOT).parts)]
    for path in files:
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if SENSITIVE.search(text):
            errors.append(f"private-material pattern in {path.relative_to(ROOT)}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if target.startswith(("https://", "http://", "#")):
                    continue
                local = (path.parent / target.split("#")[0]).resolve()
                if not local.is_relative_to(ROOT) or not local.exists():
                    errors.append(f"invalid local link in {path.relative_to(ROOT)}: {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Checked {len(files)} public files, local links and private-material patterns.")
    print("This bounded scan complements the manual publication allowlist review.")


if __name__ == "__main__":
    main()
