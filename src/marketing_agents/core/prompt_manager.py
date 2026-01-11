"""
Centralized Prompt Management System

Provides versioning, caching, rollback, and comparison capabilities for agent prompts.
Thread-safe for concurrent access.
"""

import os
import json
import shutil
import difflib
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import OrderedDict


class PromptManager:
    """
    Manages agent prompts with versioning, caching, and rollback capabilities.

    Features:
    - In-memory caching of loaded prompts
    - Automatic versioning when prompts are updated
    - Prompt comparison using difflib
    - Rollback to previous versions
    - Change logging for audit trails
    - Thread-safe operations
    """

    def __init__(self, prompts_dir: str = "config/prompts"):
        """
        Initialize the PromptManager.

        Args:
            prompts_dir: Base directory for prompt files
        """
        self.prompts_dir = Path(prompts_dir)
        self.history_dir = self.prompts_dir / "history"
        self.log_file = Path("logs/prompt_changes.log")

        # In-memory cache: {agent_id: prompt_content}
        self._cache: Dict[str, str] = {}

        # Thread lock for thread-safe operations
        self._lock = threading.RLock()

        # Create necessary directories
        self._ensure_directories()

    def _ensure_directories(self):
        """Create required directories if they don't exist."""
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def load_prompt(self, agent_id: str, version: str = "latest") -> str:
        """
        Load a prompt for the specified agent.

        Args:
            agent_id: Identifier for the agent (e.g., 'marketing_strategy')
            version: Version to load. Use 'latest' for current, or timestamp for historical

        Returns:
            The prompt content as a string

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            ValueError: If the specified version doesn't exist
        """
        with self._lock:
            # If loading latest and cached, return from cache
            if version == "latest" and agent_id in self._cache:
                return self._cache[agent_id]

            # Determine file path
            if version == "latest":
                prompt_file = self.prompts_dir / f"{agent_id}.txt"
            else:
                prompt_file = self.history_dir / f"{agent_id}_{version}.txt"

            # Check if file exists
            if not prompt_file.exists():
                if version == "latest":
                    raise FileNotFoundError(
                        f"Prompt file not found for agent '{agent_id}' at {prompt_file}"
                    )
                else:
                    raise ValueError(
                        f"Version '{version}' not found for agent '{agent_id}'"
                    )

            # Read the prompt
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_content = f.read()

            # Cache if it's the latest version
            if version == "latest":
                self._cache[agent_id] = prompt_content

            return prompt_content

    def update_prompt(self, agent_id: str, new_prompt: str, reason: str) -> bool:
        """
        Update a prompt and save the old version to history.

        Args:
            agent_id: Identifier for the agent
            new_prompt: New prompt content
            reason: Reason for the update (for logging)

        Returns:
            True if update was successful, False otherwise
        """
        with self._lock:
            try:
                prompt_file = self.prompts_dir / f"{agent_id}.txt"

                # Generate version ID (timestamp)
                version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

                # If current prompt exists, save it to history
                if prompt_file.exists():
                    history_file = self.history_dir / f"{agent_id}_{version_id}.txt"
                    shutil.copy2(prompt_file, history_file)

                # Write new prompt
                with open(prompt_file, "w", encoding="utf-8") as f:
                    f.write(new_prompt)

                # Update cache
                self._cache[agent_id] = new_prompt

                # Log the change
                self._log_prompt_change(agent_id, version_id, reason)

                return True

            except Exception as e:
                # Log the error
                self._log_prompt_change(
                    agent_id, version_id, f"ERROR: {reason} - {str(e)}"
                )
                return False

    def rollback_prompt(self, agent_id: str, version_id: str) -> bool:
        """
        Rollback a prompt to a previous version.

        Args:
            agent_id: Identifier for the agent
            version_id: Version timestamp to rollback to (e.g., '20260110_143052')

        Returns:
            True if rollback was successful, False otherwise
        """
        with self._lock:
            try:
                # Load the historical version
                old_prompt = self.load_prompt(agent_id, version_id)

                # Update with the old prompt
                success = self.update_prompt(
                    agent_id, old_prompt, f"Rollback to version {version_id}"
                )

                return success

            except Exception as e:
                self._log_prompt_change(
                    agent_id, version_id, f"ERROR: Rollback failed - {str(e)}"
                )
                return False

    def compare_prompts(
        self, agent_id: str, version1: str, version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a prompt.

        Args:
            agent_id: Identifier for the agent
            version1: First version (timestamp or 'latest')
            version2: Second version (timestamp or 'latest')

        Returns:
            Dictionary containing:
                - version1: First version identifier
                - version2: Second version identifier
                - diff_html: HTML diff output
                - diff_unified: Unified diff format
                - changes_summary: Summary of changes
        """
        with self._lock:
            try:
                # Load both versions
                prompt1 = self.load_prompt(agent_id, version1)
                prompt2 = self.load_prompt(agent_id, version2)

                # Split into lines for comparison
                lines1 = prompt1.splitlines(keepends=True)
                lines2 = prompt2.splitlines(keepends=True)

                # Generate unified diff
                diff_unified = list(
                    difflib.unified_diff(
                        lines1,
                        lines2,
                        fromfile=f"{agent_id}_{version1}",
                        tofile=f"{agent_id}_{version2}",
                        lineterm="",
                    )
                )

                # Generate HTML diff
                differ = difflib.HtmlDiff()
                diff_html = differ.make_file(
                    lines1,
                    lines2,
                    fromdesc=f"{agent_id}_{version1}",
                    todesc=f"{agent_id}_{version2}",
                )

                # Calculate changes summary
                added_lines = sum(
                    1
                    for line in diff_unified
                    if line.startswith("+") and not line.startswith("+++")
                )
                removed_lines = sum(
                    1
                    for line in diff_unified
                    if line.startswith("-") and not line.startswith("---")
                )

                return {
                    "version1": version1,
                    "version2": version2,
                    "diff_html": diff_html,
                    "diff_unified": "\n".join(diff_unified),
                    "changes_summary": {
                        "added_lines": added_lines,
                        "removed_lines": removed_lines,
                        "total_changes": added_lines + removed_lines,
                    },
                }

            except Exception as e:
                return {"error": str(e), "version1": version1, "version2": version2}

    def list_versions(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        List all available versions for an agent.

        Args:
            agent_id: Identifier for the agent

        Returns:
            List of version dictionaries with metadata:
                - version_id: Version timestamp
                - file_path: Path to the version file
                - size: File size in bytes
                - modified: Last modified timestamp
                - is_current: True if this is the current version
        """
        with self._lock:
            versions = []

            # Check for current version
            current_file = self.prompts_dir / f"{agent_id}.txt"
            if current_file.exists():
                stat = current_file.stat()
                versions.append(
                    {
                        "version_id": "latest",
                        "file_path": str(current_file),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_current": True,
                    }
                )

            # List historical versions
            pattern = f"{agent_id}_*.txt"
            for version_file in sorted(self.history_dir.glob(pattern), reverse=True):
                # Extract version ID from filename
                version_id = version_file.stem.replace(f"{agent_id}_", "")
                stat = version_file.stat()

                versions.append(
                    {
                        "version_id": version_id,
                        "file_path": str(version_file),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_current": False,
                    }
                )

            return versions

    def _log_prompt_change(self, agent_id: str, version_id: str, reason: str):
        """
        Log a prompt change to the audit log.

        Args:
            agent_id: Identifier for the agent
            version_id: Version timestamp
            reason: Reason for the change
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "version_id": version_id,
            "reason": reason,
        }

        # Append to log file (JSON lines format)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def clear_cache(self):
        """Clear the in-memory prompt cache."""
        with self._lock:
            self._cache.clear()

    def reload_prompt(self, agent_id: str) -> str:
        """
        Force reload a prompt from disk, bypassing cache.

        Args:
            agent_id: Identifier for the agent

        Returns:
            The reloaded prompt content
        """
        with self._lock:
            # Remove from cache if present
            if agent_id in self._cache:
                del self._cache[agent_id]

            # Load fresh from disk (will re-cache)
            return self.load_prompt(agent_id)

    def get_change_history(
        self, agent_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the change history from the log.

        Args:
            agent_id: Optional filter by agent ID
            limit: Maximum number of entries to return

        Returns:
            List of log entries (most recent first)
        """
        if not self.log_file.exists():
            return []

        entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if agent_id is None or entry.get("agent_id") == agent_id:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue

        # Return most recent first
        return list(reversed(entries))[:limit]
