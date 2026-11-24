"""Shared socket directory for the editor and the read-only API bind mount."""
import os
from pathlib import Path


def bridge_directory():
    override = os.getenv("GAKUCHO_UE_BRIDGE_DIR")
    if override:
        path = Path(override).resolve()
    else:
        # Only the owner can reach the host directory. The container sees just
        # the inner mount, where its different numeric UID can use the socket.
        parent = Path("/tmp/gakucho-ai-motion-ue")
        parent.mkdir(mode=0o700, exist_ok=True)
        info = parent.lstat()
        if parent.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError(f"{parent} must be a private directory owned by this user")
        path = parent / "bridge"
    path.mkdir(exist_ok=True, parents=True)
    return path


if __name__ == "__main__":
    print(bridge_directory())
