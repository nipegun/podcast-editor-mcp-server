"""Shared state for the MCP server.

The pipeline, the configuration and the background job executor are built
lazily on first use, never at import time: importing ``podcast_editor_mcp_server.server``
must not create directories or load models as a side effect.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from podcast_editor_mcp_server.config import AppContext
from podcast_editor_mcp_server.errors import JobError
from podcast_editor_mcp_server.logging_setup import fGetLogger
from podcast_editor_mcp_server.pipeline import PodcastPipeline

vLog = fGetLogger("mcp.state")


class ServerState:
  """Everything the tools share, guarded for concurrent requests."""

  def __init__(self) -> None:
    self._vContext: AppContext | None = None
    self._vPipeline: PodcastPipeline | None = None
    self._vLock = threading.RLock()
    self._vExecutor: ThreadPoolExecutor | None = None
    self._vClosing = False
    self._sCancelled: set[str] = set()
    self._dBackgroundJobs: dict[str, Future[Any]] = {}

  # ------------------------------------------------------------- lifecycle --

  def mGetContext(self) -> AppContext:
    with self._vLock:
      if self._vContext is None:
        self._vContext = AppContext()
      return self._vContext

  context = property(mGetContext)

  def mGetPipeline(self) -> PodcastPipeline:
    with self._vLock:
      if self._vPipeline is None:
        self._vPipeline = PodcastPipeline(self.context)
        self._mPurgeOldJobs(self._vPipeline)
      return self._vPipeline

  pipeline = property(mGetPipeline)

  def mGetExecutor(self) -> ThreadPoolExecutor:
    with self._vLock:
      if self._vClosing:
        raise JobError("The server is shutting down; no new background jobs can be accepted")
      if self._vExecutor is None:
        vWorkers = max(1, self.context.config.jobs.max_concurrent)
        self._vExecutor = ThreadPoolExecutor(max_workers=vWorkers, thread_name_prefix="podcast-editor-mcp-server-job")
      return self._vExecutor

  executor = property(mGetExecutor)

  def mGetAdmissionLock(self) -> threading.RLock:
    """Held while a background job is checked, recorded and submitted, so two
    requests cannot both take the last free slot."""
    return self._vLock

  admission_lock = property(mGetAdmissionLock)

  def _mPurgeOldJobs(self, pPipeline: PodcastPipeline) -> None:
    """Apply ``jobs.retention_days`` once per configuration, as the CLI purge does."""
    try:
      vRemoved = pPipeline.jobs.mPurgeOlderThan(self.context.config.jobs.retention_days)
    except OSError as vError:
      vLog.warning("Could not purge old job records", extra={"reason": str(vError)})
      return
    if vRemoved:
      vLog.info("Purged old job records", extra={"removed": vRemoved})

  def mReload(self) -> None:
    """Drop cached configuration so the next call re-reads it from disk."""
    with self._vLock:
      self._vContext = None
      self._vPipeline = None

  def mShutdown(self) -> None:
    """Stop the background executor, waiting for running jobs."""
    with self._vLock:
      self._vClosing = True
      vExecutor = self._vExecutor
      self._vExecutor = None
    # Workers and cancellation callbacks still need the state lock while
    # finishing. Never hold it across a blocking join.
    if vExecutor is not None:
      vExecutor.shutdown(wait=True, cancel_futures=True)

  # ---------------------------------------------------------- cancellation --

  def mRequestCancel(self, pJobId: str) -> None:
    with self._vLock:
      self._sCancelled.add(pJobId)
      vFuture = self._dBackgroundJobs.get(pJobId)
    if vFuture is not None:
      vFuture.cancel()

  def mIsCancelled(self, pJobId: str) -> bool:
    with self._vLock:
      return pJobId in self._sCancelled

  def mClearCancel(self, pJobId: str) -> None:
    with self._vLock:
      self._sCancelled.discard(pJobId)

  # ------------------------------------------------------- background jobs --

  def mCheckBackgroundCapacity(self) -> None:
    """Refuse a new background job when the running and waiting ones fill every slot."""
    with self._vLock:
      vSettings = self.context.config.jobs
      vLimit = max(1, vSettings.max_concurrent) + vSettings.max_queued
      vActive = sum(1 for vFuture in self._dBackgroundJobs.values() if not vFuture.done())
      if vActive >= vLimit:
        raise JobError(
          "Too many background jobs are running or waiting; try again when one finishes",
          {"active": vActive, "limit": vLimit, "retryable": True},
        )

  def mRegisterBackground(self, pJobId: str, pFuture: Future[Any]) -> None:
    with self._vLock:
      self._dBackgroundJobs[pJobId] = pFuture
    # A finished job leaves the table at once. Registered first, so a future
    # that is already done cannot be removed before it is added.
    pFuture.add_done_callback(lambda pDone: self._mForgetBackground(pJobId, pDone))

  def _mForgetBackground(self, pJobId: str, pFuture: Future[Any]) -> None:
    with self._vLock:
      if self._dBackgroundJobs.get(pJobId) is pFuture:
        self._dBackgroundJobs.pop(pJobId, None)

  def mBackgroundJobIds(self) -> list[str]:
    with self._vLock:
      return [vKey for vKey, vFuture in self._dBackgroundJobs.items() if not vFuture.done()]
