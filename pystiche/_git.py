import subprocess
from os import path
from typing import Optional

__all__ = ["is_available", "is_repo", "is_dirty", "hash"]


def _run(*cmds: str, cwd: Optional[str] = None) -> str:
    return subprocess.check_output(("git", *cmds), cwd=cwd).decode("utf-8").strip()


def is_available() -> bool:
    try:
        _run("--help")
        return True
    except subprocess.CalledProcessError:
        return False
    except OSError:
        return False


def is_repo(dir: str) -> bool:
    return path.exists(path.join(dir, ".git"))


def is_dirty(dir: str):
    return bool(_run("status", "-uno", "--porcelain", cwd=dir))


def hash(dir: str):
    return _run("rev-parse", "--short", "HEAD", cwd=dir)
