"""
Test and demonstration script for PromptManager.

This script demonstrates:
1. Loading prompts
2. Updating prompts with versioning
3. Comparing versions
4. Rolling back to previous versions
5. Viewing change history
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.core.prompt_manager import PromptManager


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    """Demonstrate PromptManager functionality."""

    # Initialize PromptManager
    print_section("1. Initialize PromptManager")
    prompt_manager = PromptManager(prompts_dir="config/prompts")
    print("✓ PromptManager initialized")

    # Load a prompt
    print_section("2. Load Marketing Strategy Prompt")
    try:
        prompt = prompt_manager.load_prompt("marketing_strategy")
        print(f"✓ Loaded prompt ({len(prompt)} characters)")
        print(f"First 200 characters:\n{prompt[:200]}...\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    # List current versions
    print_section("3. List Current Versions")
    versions = prompt_manager.list_versions("marketing_strategy")
    print(f"Found {len(versions)} version(s):")
    for v in versions:
        print(
            f"  - {v['version_id']}: {v['size']} bytes, "
            f"modified {v['modified']}, current={v['is_current']}"
        )

    # Update prompt (create a version)
    print_section("4. Update Prompt (Creates Version)")
    new_prompt = (
        prompt
        + "\n\n## BUDGET JUSTIFICATION\n\nAll strategies must include detailed budget justification."
    )
    success = prompt_manager.update_prompt(
        "marketing_strategy", new_prompt, "Added budget justification requirements"
    )
    if success:
        print("✓ Prompt updated successfully")
    else:
        print("✗ Failed to update prompt")

    # List versions after update
    print_section("5. List Versions After Update")
    versions = prompt_manager.list_versions("marketing_strategy")
    print(f"Found {len(versions)} version(s):")
    for v in versions:
        print(
            f"  - {v['version_id']}: {v['size']} bytes, "
            f"modified {v['modified']}, current={v['is_current']}"
        )

    # Compare versions (if we have at least 2)
    if len(versions) >= 2:
        print_section("6. Compare Versions")
        version1 = versions[1]["version_id"]  # First historical version
        version2 = "latest"

        comparison = prompt_manager.compare_prompts(
            "marketing_strategy", version1, version2
        )

        if "error" not in comparison:
            summary = comparison["changes_summary"]
            print(f"Comparison: {version1} vs {version2}")
            print(f"  Added lines: {summary['added_lines']}")
            print(f"  Removed lines: {summary['removed_lines']}")
            print(f"  Total changes: {summary['total_changes']}")
            print(f"\nUnified diff preview:")
            diff_lines = comparison["diff_unified"].split("\n")
            for line in diff_lines[:20]:  # Show first 20 lines
                print(f"  {line}")
        else:
            print(f"✗ Comparison error: {comparison['error']}")

    # View change history
    print_section("7. View Change History")
    history = prompt_manager.get_change_history("marketing_strategy", limit=5)
    print(f"Recent changes ({len(history)} entries):")
    for entry in history:
        print(f"  [{entry['timestamp']}] {entry['version_id']}")
        print(f"    Reason: {entry['reason']}")

    # Demonstrate rollback (if we have versions to rollback to)
    if len(versions) >= 2:
        print_section("8. Rollback to Previous Version")
        version_to_restore = versions[1]["version_id"]
        print(f"Rolling back to version: {version_to_restore}")

        success = prompt_manager.rollback_prompt(
            "marketing_strategy", version_to_restore
        )

        if success:
            print("✓ Rollback successful")

            # Verify rollback
            restored_prompt = prompt_manager.load_prompt("marketing_strategy")
            print(f"✓ Restored prompt ({len(restored_prompt)} characters)")
        else:
            print("✗ Rollback failed")

    # Cache demonstration
    print_section("9. Cache Demonstration")
    print("Loading prompt multiple times...")
    import time

    # Clear cache first
    prompt_manager.clear_cache()

    # First load (from disk)
    start = time.time()
    prompt_manager.load_prompt("marketing_strategy")
    first_load = time.time() - start

    # Second load (from cache)
    start = time.time()
    prompt_manager.load_prompt("marketing_strategy")
    cached_load = time.time() - start

    print(f"  First load (disk): {first_load*1000:.2f}ms")
    print(f"  Cached load: {cached_load*1000:.2f}ms")
    print(f"  Speedup: {first_load/cached_load:.1f}x faster")

    print_section("Demo Complete")
    print("PromptManager is ready for use!")
    print("\nNext steps:")
    print("  - Integrate with BaseAgent")
    print("  - Add API endpoints for prompt management")
    print("  - Set up automated prompt testing")


if __name__ == "__main__":
    main()
