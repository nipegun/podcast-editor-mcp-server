"""Workspace path confinement.

Every path that arrives from an MCP client, a CLI argument or an EditPlan is
funnelled through :class:`PathGuard`. Nothing else in the product is allowed to
call ``open()`` on a caller-supplied string.

The rules are deliberately blunt:

* A path is resolved with :meth:`pathlib.Path.resolve`, which collapses ``..``
  and follows symlinks. The *resolved* path must sit inside one of the
  configured roots, so a symlink pointing outside the workspace resolves
  outside and is rejected.
* Unless symlinks are explicitly allowed, no component of the path inside the
  root may be a symlink at all, which closes the "swap the link after the
  check" window as far as a single process can.
* File names we create are sanitised and length-limited, and extensions are
  checked against per-operation allow lists.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from podcast_editor_mcp_server.errors import SecurityError
from podcast_editor_mcp_server.logging_setup import fGetLogger

vLog = fGetLogger("security.paths")

cMaxNameLength: int = 200
cMaxPathLength: int = 3000

# Control characters, path separators and shell-hostile characters are refused
# in file names we create. Names that merely *exist* on disk are checked less
# strictly, because we did not choose them.
cForbiddenNameChars = re.compile(r"[\x00-\x1f\x7f/\\]")

cAudioReadExtensions: frozenset[str] = frozenset(
  {".wav", ".wave", ".flac", ".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".oga", ".opus", ".wv", ".aiff", ".aif", ".caf"}
)
cAudioWriteExtensions: frozenset[str] = frozenset({".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"})
cDataExtensions: frozenset[str] = frozenset({".json", ".txt", ".srt", ".vtt", ".yaml", ".yml", ".log", ".csv"})


@dataclass(frozen=True, slots=True)
class WorkspaceRoot:
  """One authorised directory, identified by a stable role name."""

  role: str
  path: Path
  writable: bool


class PathGuard:
  """Confines filesystem access to a set of authorised roots."""

  def __init__(self, pRoots: list[WorkspaceRoot], pAllowSymlinks: bool = False) -> None:
    if not pRoots:
      raise SecurityError("PathGuard requires at least one authorised root")
    lResolved: list[WorkspaceRoot] = []
    for vRoot in pRoots:
      vResolvedPath = vRoot.path.expanduser().resolve()
      lResolved.append(WorkspaceRoot(role=vRoot.role, path=vResolvedPath, writable=vRoot.writable))
    self._aRoots: tuple[WorkspaceRoot, ...] = tuple(lResolved)
    self._vAllowSymlinks = bool(pAllowSymlinks)

  # ----------------------------------------------------------------- roots --

  def mGetRoots(self) -> tuple[WorkspaceRoot, ...]:
    return self._aRoots

  roots = property(mGetRoots)

  def mRootFor(self, pRole: str) -> Path:
    """Return the directory registered under ``pRole``."""
    for vRoot in self._aRoots:
      if vRoot.role == pRole:
        return vRoot.path
    raise SecurityError(f"Unknown workspace role: {pRole}", {"known_roles": [vR.role for vR in self._aRoots]})

  def mHasRole(self, pRole: str) -> bool:
    return any(vRoot.role == pRole for vRoot in self._aRoots)

  def mEnsureRoots(self) -> None:
    """Create every configured root directory if it does not exist yet."""
    for vRoot in self._aRoots:
      vRoot.path.mkdir(parents=True, exist_ok=True)
  # ------------------------------------------------------------ resolution --

  def mIsInside(self, pPath: Path) -> bool:
    """Whether an already-resolved path sits inside any authorised root."""
    return any(pPath == vRoot.path or vRoot.path in pPath.parents for vRoot in self._aRoots)

  def mOwningRoot(self, pPath: Path) -> WorkspaceRoot | None:
    """Return the innermost root that contains ``pPath``."""
    vBest: WorkspaceRoot | None = None
    for vRoot in self._aRoots:
      vContains = pPath == vRoot.path or vRoot.path in pPath.parents
      if vContains and (vBest is None or len(vRoot.path.parts) > len(vBest.path.parts)):
        vBest = vRoot
    return vBest

  def mLabel(self, pPath: Path) -> str:
    """Return a short ``role/relative`` label, used in reports and logs."""
    vRoot = self.mOwningRoot(pPath)
    if vRoot is None:
      return pPath.name
    if pPath == vRoot.path:
      return vRoot.role
    return f"{vRoot.role}/{pPath.relative_to(vRoot.path)}"

  def mResolve(
    self,
    pCandidate: str | Path,
    pDefaultRole: str | None = None,
    pRequireExists: bool = True,
    pForWrite: bool = False,
    pAllowedExtensions: frozenset[str] | None = None,
    pMustBeFile: bool = True,
  ) -> Path:
    """Resolve a caller-supplied path and confine it to the workspace."""
    vRaw = str(pCandidate)
    if not vRaw.strip():
      raise SecurityError("Empty path is not allowed")
    if "\x00" in vRaw:
      raise SecurityError("Path contains a NUL byte")
    if len(vRaw) > cMaxPathLength:
      raise SecurityError("Path is too long", {"max_length": cMaxPathLength})

    # Normalise so that visually identical but differently encoded names cannot
    # be used to sneak past the checks below.
    vRaw = unicodedata.normalize("NFC", vRaw)

    vPath = Path(vRaw)
    if vPath.is_absolute():
      vCandidate = vPath
    else:
      vCandidate = self._mResolveRelative(vPath, pDefaultRole, pRequireExists)

    # expanduser() is applied only to the leading "~"; it can never introduce a
    # traversal because resolve() runs afterwards and the result is re-checked.
    vResolved = vCandidate.expanduser().resolve()

    if not self.mIsInside(vResolved):
      vLog.warning(
        "Rejected path outside the workspace",
        extra={"requested": vRaw, "resolved": str(vResolved)},
      )
      raise SecurityError(
        "Path is outside the authorised workspace",
        {"requested": vRaw, "allowed_roots": [str(vRoot.path) for vRoot in self._aRoots]},
      )

    if not self._vAllowSymlinks:
      # Check the path as it was *asked for*, not only as it resolved: a link
      # whose target happens to sit inside the workspace resolves cleanly and
      # would otherwise slip through. normpath collapses ".." without touching
      # the filesystem, so it cannot itself follow a link.
      self._mAssertNoSymlinkComponents(Path(os.path.normpath(vCandidate)))
      self._mAssertNoSymlinkComponents(vResolved)

    if pForWrite:
      vRoot = self.mOwningRoot(vResolved)
      if vRoot is None or not vRoot.writable:
        raise SecurityError(
          "Destination is inside a read-only workspace root",
          {"path": self.mLabel(vResolved)},
        )

    if pRequireExists:
      if not vResolved.exists():
        raise SecurityError("Path does not exist", {"path": self.mLabel(vResolved)})
      if pMustBeFile and not vResolved.is_file():
        raise SecurityError("Path is not a regular file", {"path": self.mLabel(vResolved)})

    if pAllowedExtensions is not None:
      vSuffix = vResolved.suffix.lower()
      if vSuffix not in pAllowedExtensions:
        raise SecurityError(
          "File extension is not allowed for this operation",
          {"extension": vSuffix or "<none>", "allowed": sorted(pAllowedExtensions)},
        )

    return vResolved

  def mResolveInput(self, pCandidate: str | Path, pDefaultRole: str = "input") -> Path:
    """Resolve an existing audio file that will only ever be read."""
    return self.mResolve(
      pCandidate,
      pDefaultRole=pDefaultRole,
      pRequireExists=True,
      pForWrite=False,
      pAllowedExtensions=cAudioReadExtensions,
    )

  def mResolveOutput(
    self,
    pCandidate: str | Path,
    pDefaultRole: str = "output",
    pAllowedExtensions: frozenset[str] | None = None,
    pOverwrite: bool = False,
    pSource: Path | None = None,
    pCreateParent: bool = True,
  ) -> Path:
    """Resolve a destination path, creating its parent directory."""
    vResolved = self.mResolve(
      pCandidate,
      pDefaultRole=pDefaultRole,
      pRequireExists=False,
      pForWrite=True,
      pAllowedExtensions=pAllowedExtensions,
      pMustBeFile=False,
    )
    if pSource is not None and (
      vResolved == pSource.resolve()
      or (vResolved.exists() and pSource.exists() and vResolved.samefile(pSource))
    ):
      raise SecurityError("Destination would overwrite the source recording", {"path": self.mLabel(vResolved)})
    if vResolved.exists():
      if vResolved.is_dir():
        raise SecurityError("Destination is a directory", {"path": self.mLabel(vResolved)})
      if not pOverwrite:
        raise SecurityError(
          "Destination already exists; pass overwrite to replace it",
          {"path": self.mLabel(vResolved)},
        )
    if pCreateParent:
      vResolved.parent.mkdir(parents=True, exist_ok=True)
    return vResolved
  # -------------------------------------------------------------- internals --

  def _mResolveRelative(self, pPath: Path, pDefaultRole: str | None, pRequireExists: bool) -> Path:
    """Turn a relative path into an absolute one inside the workspace.

    Two spellings are accepted, and both are safe because the result is
    re-checked against the roots afterwards:

    * ``episode.wav`` - relative to ``pDefaultRole``;
    * ``output/Podcast-042/Podcast-042.edit-plan.json`` - prefixed with a role
      name, which is exactly the label the tools hand back. Without this, a
      caller could not feed a returned label straight back in.

    When both readings are possible the existing file wins, so an unlucky
    directory name inside ``input`` cannot hijack a plain relative path.
    """
    vParts = pPath.parts
    vRoleCandidate: Path | None = None
    if vParts and self.mHasRole(vParts[0]):
      vRoleCandidate = self.mRootFor(vParts[0]).joinpath(*vParts[1:]) if len(vParts) > 1 else self.mRootFor(vParts[0])

    vDefaultCandidate: Path | None = None
    if pDefaultRole is not None:
      vDefaultCandidate = self.mRootFor(pDefaultRole) / pPath

    if vRoleCandidate is None and vDefaultCandidate is None:
      raise SecurityError("Relative paths need a default workspace role", {"path": str(pPath)})
    if vRoleCandidate is None:
      return vDefaultCandidate  # type: ignore[return-value]
    if vDefaultCandidate is None:
      return vRoleCandidate

    if pRequireExists:
      if vDefaultCandidate.exists():
        return vDefaultCandidate
      if vRoleCandidate.exists():
        return vRoleCandidate
      return vDefaultCandidate
    return vRoleCandidate

  def _mAssertNoSymlinkComponents(self, pResolved: Path) -> None:
    """Refuse a path whose components inside the root include a symlink.

    ``pResolved`` is already symlink-free by construction, so this walks the
    *original* locations: for each root that contains the path we re-check the
    unresolved chain, which catches a link that resolve() silently followed
    back into the workspace.
    """
    vRoot = self.mOwningRoot(pResolved)
    if vRoot is None:
      return
    vCurrent = vRoot.path
    try:
      vRelative = pResolved.relative_to(vRoot.path)
    except ValueError:
      return
    for vPart in vRelative.parts:
      vCurrent = vCurrent / vPart
      if vCurrent.is_symlink():
        vLog.warning("Rejected symlinked path component", extra={"component": str(vCurrent)})
        raise SecurityError(
          "Symlinks are not allowed inside the workspace",
          {"component": self.mLabel(vCurrent)},
        )


def fSanitiseName(pName: str, pFallback: str = "podcast") -> str:
  """Turn arbitrary text into a safe file-name stem.

  Used for artifact directories derived from a source file name. The result is
  never empty, never starts with a dot or dash, and contains no separators.
  """
  vName = unicodedata.normalize("NFC", str(pName)).strip()
  vName = cForbiddenNameChars.sub("_", vName)
  vName = vName.replace("‮", "_")  # right-to-left override
  vName = vName.strip(" ._")
  while vName.startswith("-"):
    vName = vName[1:]
  if len(vName) > cMaxNameLength:
    vName = vName[:cMaxNameLength].rstrip(" .")
  if not vName:
    return pFallback
  return vName
