"""Content hashing used for provenance and cache keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

cChunkBytes: int = 1024 * 1024


def fHashFile(pPath: Path, pAlgorithm: str = "sha256") -> str:
  """Return the hex digest of a file, read in chunks so size does not matter."""
  vHasher = hashlib.new(pAlgorithm)
  with pPath.open("rb") as vHandle:
    while True:
      vChunk = vHandle.read(cChunkBytes)
      if not vChunk:
        break
      vHasher.update(vChunk)
  return vHasher.hexdigest()


def fHashBytes(pData: bytes, pAlgorithm: str = "sha256") -> str:
  """Return the hex digest of a byte string."""
  return hashlib.new(pAlgorithm, pData).hexdigest()


def fHashText(pText: str, pAlgorithm: str = "sha256") -> str:
  """Return the hex digest of a UTF-8 encoded string."""
  return fHashBytes(pText.encode("utf-8"), pAlgorithm)


def fHashMapping(pMapping: dict[str, Any], pAlgorithm: str = "sha256") -> str:
  """Hash a JSON-serialisable mapping deterministically.

  Keys are sorted and floats are written with ``repr`` semantics by
  ``json.dumps``, so the same logical configuration always yields the same
  digest across runs and machines.
  """
  vCanonical = json.dumps(pMapping, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
  return fHashText(vCanonical, pAlgorithm)


def fShortHash(pDigest: str, pLength: int = 12) -> str:
  """Shorten a digest for use inside file names."""
  return pDigest[:pLength]
