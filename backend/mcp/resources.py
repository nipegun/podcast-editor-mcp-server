"""MCP resources.

Resources expose *documents the client may want to read*, not the filesystem.
Every URI is anchored to a known product object - a profile, a job, an
artifact of a job - and nothing outside the workspace is reachable through
them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError

from podcast_editor_mcp_server.config import fListProfiles, fLoadProfile
from podcast_editor_mcp_server.errors import PodcastEditorMcpServerError
from podcast_editor_mcp_server.jobs.store import fValidateJobId
from podcast_editor_mcp_server.mcp.naming import fMcpParameters
from podcast_editor_mcp_server.mcp.state import ServerState
from podcast_editor_mcp_server.models.edit_plan import cMasteringTypes, cStructuralTypes
from podcast_editor_mcp_server.version import cProductName, cVersion

cMaxResourceBytes: int = 4 * 1024 * 1024

# Job artifacts a client may read through a resource URI, mapped to the
# artifact ``kind`` recorded on the job.
cReadableArtifacts: dict[str, str] = {
  "transcript": "transcript_json",
  "analysis": "analysis",
  "plan": "plan",
  "report": "report",
  "candidates": "candidates",
  "simulation": "simulation",
}


def fRegisterResources(pServer: MCPServer, pState: ServerState) -> None:
  """Register every resource on ``pServer``."""

  @pServer.resource(
    "podcast-editor-mcp-server://workspace",
    name="workspace",
    title="Workspace layout",
    description="The directories this server is allowed to read from and write to.",
    mime_type="application/json",
  )
  def fWorkspaceResource() -> str:
    return _fDump(pState.context.workspace.mDescribe())

  @pServer.resource(
    "podcast-editor-mcp-server://operations",
    name="operations",
    title="Supported edit operations",
    description="Every EditPlan operation this build can execute, with its parameters.",
    mime_type="application/json",
  )
  def fOperationsResource() -> str:
    from podcast_editor_mcp_server.models.edit_plan import EditPlan

    dSchema = EditPlan.model_json_schema()
    return _fDump(
      {
        "structural": sorted(cStructuralTypes),
        "mastering": sorted(cMasteringTypes),
        "not_implemented": ["equalize", "de_esser", "move_clip", "insert_music", "duck_music"],
        "schema": dSchema,
      }
    )

  @pServer.resource(
    "podcast-editor-mcp-server://profiles",
    name="profiles",
    title="Available profiles",
    description="Every editing profile this installation can use.",
    mime_type="application/json",
  )
  def fProfilesResource() -> str:
    ldProfiles = fListProfiles(pState.context.config)
    return _fDump(
      {
        "active": pState.context.profile.name,
        "count": len(ldProfiles),
        "profiles": ldProfiles,
      }
    )

  @pServer.resource(
    "podcast-editor-mcp-server://profiles/{name}",
    name="profile",
    title="One profile",
    description="The full, validated contents of a single editing profile.",
    mime_type="application/json",
  )
  @fMcpParameters()
  def fProfileResource(pName: str) -> str:
    try:
      vProfile = fLoadProfile(pName, pState.context.config)
    except PodcastEditorMcpServerError as vError:
      raise ResourceNotFoundError(f"Profile '{pName}' was not found: {vError.message}") from vError
    return _fDump(vProfile.model_dump(mode="json"))

  @pServer.resource(
    "podcast-editor-mcp-server://jobs",
    name="jobs",
    title="Recent jobs",
    description="The most recent processing jobs and their state.",
    mime_type="application/json",
  )
  def fJobsResource() -> str:
    lJobs = pState.pipeline.jobs.mList(pLimit=50)
    return _fDump({"count": len(lJobs), "jobs": [vJob.mSummary() for vJob in lJobs]})

  @pServer.resource(
    "podcast-editor-mcp-server://jobs/{job_id}",
    name="job",
    title="One job",
    description="The full record of one processing job.",
    mime_type="application/json",
  )
  @fMcpParameters()
  def fJobResource(pJobId: str) -> str:
    vJob = _fLoadJob(pState, pJobId)
    return _fDump(vJob.model_dump(mode="json"))

  @pServer.resource(
    "podcast-editor-mcp-server://jobs/{job_id}/{artifact}",
    name="job_artifact",
    title="A job artifact",
    description="One JSON artifact of a job: transcript, analysis, plan, report, candidates or simulation.",
    mime_type="application/json",
  )
  @fMcpParameters()
  def fJobArtifactResource(pJobId: str, pArtifact: str) -> str:
    vJob = _fLoadJob(pState, pJobId)
    vKind = cReadableArtifacts.get(pArtifact)
    if vKind is None:
      raise ResourceNotFoundError(
        f"Unknown artifact '{pArtifact}'. Available: {', '.join(sorted(cReadableArtifacts))}"
      )
    for vArtifact in vJob.artifacts:
      if vArtifact.kind != vKind:
        continue
      vPath = Path(vArtifact.path)
      if not pState.pipeline.guard.mIsInside(vPath):
        raise ResourceError("That artifact is no longer inside the workspace")
      if not vPath.is_file():
        raise ResourceNotFoundError(f"The '{pArtifact}' artifact of job {pJobId} is no longer on disk")
      if vPath.stat().st_size > cMaxResourceBytes:
        raise ResourceError(
          f"The '{pArtifact}' artifact is {vPath.stat().st_size // 1024} KiB, "
          f"above the {cMaxResourceBytes // 1024} KiB resource limit; use the paginated tools instead"
        )
      return vPath.read_text(encoding="utf-8")
    raise ResourceNotFoundError(f"Job {pJobId} has no '{pArtifact}' artifact")

  @pServer.resource(
    "podcast-editor-mcp-server://server",
    name="server",
    title="Server information",
    description="Version, transport and the guarantees this server makes.",
    mime_type="application/json",
  )
  def fServerResource() -> str:
    return _fDump(
      {
        "product": cProductName,
        "version": cVersion,
        "transport": "stdio",
        "guarantees": [
          "The source audio file is opened read-only and never modified.",
          "Every path is confined to the configured workspace roots.",
          "Every edit plan is validated before execution.",
          "No audio, transcript or metadata leaves this machine.",
        ],
      }
    )


def _fDump(pPayload: Any) -> str:
  return json.dumps(pPayload, indent=2, ensure_ascii=False, default=str)


def _fLoadJob(pState: ServerState, pJobId: str) -> Any:
  try:
    fValidateJobId(pJobId)
    return pState.pipeline.jobs.mLoad(pJobId)
  except PodcastEditorMcpServerError as vError:
    raise ResourceNotFoundError(f"Job '{pJobId}' was not found: {vError.message}") from vError
