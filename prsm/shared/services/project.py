"""Project manager — directory-aware state management.

Maps the current working directory to a project-specific storage path
under ~/.prsm/projects/, with git worktree awareness for shared repository
identity across multiple working trees.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class WorktreeInfo:
    """Information about a git worktree."""
    path: str
    head: str
    branch: Optional[str] = None
    detached: bool = False
    bare: bool = False
    locked: bool = False
    lock_reason: Optional[str] = None


@dataclass
class RepositoryContext:
    """Complete context about the current working directory's git state."""
    is_git_repo: bool
    is_worktree: bool
    repo_identity: str  # Stable across worktrees (git-common-dir based)
    worktree_root: Optional[Path] = None
    branch: Optional[str] = None
    common_dir: Optional[Path] = None
    all_worktrees: list[WorktreeInfo] = None

    def __post_init__(self):
        if self.all_worktrees is None:
            self.all_worktrees = []


class ProjectManager:
    """Maps working directories to per-project storage paths with git worktree awareness."""

    @staticmethod
    def get_project_dir(cwd: Path | None = None) -> Path:
        """Map cwd to ~/.prsm/projects/{PROJECT_ID}/.

        For git repositories, uses git-common-dir for stable identity across worktrees.
        For non-git directories, uses path-based identification.
        Example: /home/user/myproject → home-user-myproject
        """
        repo_id = ProjectManager.get_repo_identity(cwd)
        project_dir = Path.home() / ".prsm" / "projects" / repo_id
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir

    @staticmethod
    def get_repo_identity(cwd: Path | None = None) -> str:
        """Get a stable identity for the repository.

        For git repos, uses git-common-dir so all worktrees share the same identity.
        For non-git directories, falls back to path-based identity.

        Returns:
            A string suitable for use as a directory name (no slashes).
        """
        common_dir = ProjectManager.get_git_common_dir(cwd)
        if common_dir:
            # Use common dir path as identity base
            # Example: /home/user/repos/myproject/.git -> home-user-repos-myproject-git
            return "-".join(common_dir.resolve().parts[1:])

        # Fallback: existing path-based identity
        resolved = (cwd or Path.cwd()).resolve()
        return "-".join(resolved.parts[1:])

    @staticmethod
    def get_memory_path(project_dir: Path) -> Path:
        """Return path to MEMORY.md for a project."""
        memory_dir = project_dir / "memory"
        memory_dir.mkdir(exist_ok=True)
        return memory_dir / "MEMORY.md"

    @staticmethod
    def get_sessions_dir(project_dir: Path) -> Path:
        """Return sessions directory for a project."""
        d = project_dir / "sessions"
        d.mkdir(exist_ok=True)
        return d

    @staticmethod
    def get_policy_dir(project_dir: Path) -> Path:
        """Return policy directory for a project."""
        d = project_dir / "policy"
        d.mkdir(exist_ok=True)
        return d

    @staticmethod
    def get_artifacts_dir(project_dir: Path) -> Path:
        """Return artifacts directory for a project."""
        d = project_dir / "artifacts"
        d.mkdir(exist_ok=True)
        return d

    @staticmethod
    def get_audit_log_path(project_dir: Path) -> Path:
        """Return audit log database path for a project."""
        audit_dir = project_dir / "audit"
        audit_dir.mkdir(exist_ok=True)
        return audit_dir / "audit.db"

    @staticmethod
    def get_git_branch(cwd: Path | None = None) -> str | None:
        """Get the current git branch, or None if not a git repo or detached HEAD."""
        try:
            result = subprocess.run(
                ["git", "symbolic-ref", "--short", "HEAD"],
                cwd=str(cwd or Path.cwd()),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    @staticmethod
    def is_git_repo(cwd: Path | None = None) -> bool:
        """Check if directory is inside a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=str(cwd or Path.cwd()),
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def get_git_common_dir(cwd: Path | None = None) -> Path | None:
        """Get the git common directory (shared across all worktrees).

        This directory contains the shared git state (refs, objects, config).
        For repositories using worktrees, this is always the main .git directory.
        For normal repos, this is the .git directory.

        Returns:
            Absolute path to the common git directory, or None if not a git repo.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                cwd=str(cwd or Path.cwd()),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass
        return None

    @staticmethod
    def get_git_dir(cwd: Path | None = None) -> Path | None:
        """Get the per-worktree git directory.

        For main worktree: same as get_git_common_dir()
        For linked worktrees: points to .git/worktrees/<name>/

        Returns:
            Absolute path to the git directory, or None if not a git repo.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--absolute-git-dir"],
                cwd=str(cwd or Path.cwd()),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass
        return None

    @staticmethod
    def is_worktree(cwd: Path | None = None) -> bool:
        """Check if current directory is a linked worktree (not the main working tree).

        Returns:
            True if this is a linked worktree, False if main worktree or not a git repo.
        """
        common = ProjectManager.get_git_common_dir(cwd)
        git_dir = ProjectManager.get_git_dir(cwd)
        return common is not None and git_dir is not None and common != git_dir

    @staticmethod
    def get_worktree_root(cwd: Path | None = None) -> Path | None:
        """Get the root of the current worktree.

        This returns the working tree root for the CURRENT worktree, which may
        differ from the main repository root if working in a linked worktree.

        Returns:
            Absolute path to the worktree root, or None if not a git repo.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=str(cwd or Path.cwd()),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass
        return None

    @staticmethod
    def list_worktrees(cwd: Path | None = None) -> list[WorktreeInfo]:
        """List all worktrees for the current repository.

        Returns:
            List of WorktreeInfo objects, or empty list if not a git repo or no worktrees.
        """
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=str(cwd or Path.cwd()),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []

            worktrees = []
            current = {}
            for line in result.stdout.splitlines():
                if not line:
                    if current:
                        worktrees.append(WorktreeInfo(**current))
                        current = {}
                    continue
                if line.startswith("worktree "):
                    current["path"] = line[len("worktree "):]
                elif line.startswith("HEAD "):
                    current["head"] = line[len("HEAD "):]
                elif line.startswith("branch "):
                    current["branch"] = line[len("branch "):]
                elif line == "bare":
                    current["bare"] = True
                elif line == "detached":
                    current["detached"] = True
                elif line.startswith("locked"):
                    current["locked"] = True
                    if " " in line:
                        current["lock_reason"] = line.split(" ", 1)[1]
            if current:
                worktrees.append(WorktreeInfo(**current))
            return worktrees
        except Exception:
            return []

    @staticmethod
    def create_worktree(
        path: str,
        branch: str | None = None,
        new_branch: str | None = None,
        cwd: Path | None = None,
    ) -> tuple[bool, str]:
        """Create a new git worktree.

        Args:
            path: Filesystem path for the new worktree.
            branch: Existing branch to check out (mutually exclusive with new_branch).
            new_branch: Create a new branch with this name at the new worktree.
            cwd: Working directory to run git from.

        Returns:
            (success, message) tuple.
        """
        cmd = ["git", "worktree", "add"]
        if new_branch:
            cmd += ["-b", new_branch, path]
        elif branch:
            cmd += [path, branch]
        else:
            # Detached HEAD at current commit
            cmd += ["--detach", path]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd or Path.cwd()),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                msg = result.stdout.strip() or result.stderr.strip() or "Worktree created"
                return True, msg
            return False, result.stderr.strip() or "Unknown error"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            return False, "git not found"
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def remove_worktree(
        path: str,
        force: bool = False,
        cwd: Path | None = None,
    ) -> tuple[bool, str]:
        """Remove a git worktree.

        Args:
            path: Path of the worktree to remove.
            force: If True, remove even if the worktree has modifications.
            cwd: Working directory to run git from.

        Returns:
            (success, message) tuple.
        """
        cmd = ["git", "worktree", "remove"]
        if force:
            cmd.append("--force")
        cmd.append(path)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd or Path.cwd()),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True, result.stdout.strip() or "Worktree removed"
            return False, result.stderr.strip() or "Unknown error"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except FileNotFoundError:
            return False, "git not found"
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def get_repository_context(cwd: Path | None = None) -> RepositoryContext:
        """Get complete repository and worktree context for the current directory.

        This is the primary method to use when you need to understand the git state.

        Returns:
            RepositoryContext with all relevant git and worktree information.
        """
        cwd = cwd or Path.cwd()
        try:
            result = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "--is-inside-work-tree",
                    "--path-format=absolute",
                    "--git-common-dir",
                    "--absolute-git-dir",
                    "--show-toplevel",
                ],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            result = None

        if result is None or result.returncode != 0:
            # Non-git directory: use path-based identity
            return RepositoryContext(
                is_git_repo=False,
                is_worktree=False,
                repo_identity=ProjectManager.get_repo_identity(cwd),
            )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) < 4 or lines[0] != "true":
            return RepositoryContext(
                is_git_repo=False,
                is_worktree=False,
                repo_identity=ProjectManager.get_repo_identity(cwd),
            )

        common_dir = Path(lines[1])
        git_dir = Path(lines[2])
        worktree_root = Path(lines[3])
        is_wt = common_dir != git_dir
        branch = ProjectManager.get_git_branch(cwd)
        all_wt = ProjectManager.list_worktrees(cwd)
        repo_identity = "-".join(common_dir.resolve().parts[1:])

        return RepositoryContext(
            is_git_repo=True,
            is_worktree=is_wt,
            repo_identity=repo_identity,
            worktree_root=worktree_root,
            branch=branch,
            common_dir=common_dir,
            all_worktrees=all_wt,
        )
