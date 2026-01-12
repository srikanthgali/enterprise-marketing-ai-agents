#!/usr/bin/env python3
"""
Validate .env file format to ensure no inline comments on assignment lines.
Inline comments can cause Pydantic validation errors.
"""
import sys
from pathlib import Path


def validate_env_file(env_path: Path) -> tuple[bool, list[str]]:
    """
    Validate .env file format.

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    if not env_path.exists():
        return False, [f"File not found: {env_path}"]

    errors = []
    with open(env_path) as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines and comment-only lines
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Check if line has an assignment and an inline comment
            if "=" in line and "#" in line:
                # Get the part after the '='
                value_part = line.split("=", 1)[1]
                if "#" in value_part:
                    errors.append(
                        f"Line {line_num}: Inline comment detected after value. "
                        f"Move comment to a separate line above.\n  {line.rstrip()}"
                    )

    return len(errors) == 0, errors


def main():
    """Main validation function."""
    # Check .env file
    env_file = Path(__file__).parent.parent / ".env"

    is_valid, errors = validate_env_file(env_file)

    if is_valid:
        print(f"✓ {env_file} format is valid")
        return 0
    else:
        print(f"✗ {env_file} has format issues:\n")
        for error in errors:
            print(error)
        print("\nInline comments on assignment lines cause Pydantic validation errors.")
        print("Move comments to separate lines above the assignments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
