"""MCP prompts.

These are workflow templates, not magic. They tell a client model which tools
exist, in which order they make sense, and where the product deliberately
refuses to decide on its own.
"""

from __future__ import annotations

from mcp.server import MCPServer

from podcast_editor_mcp_server.mcp.naming import fMcpParameters
from podcast_editor_mcp_server.mcp.state import ServerState


def fRegisterPrompts(pServer: MCPServer, pState: ServerState) -> None:
  """Register every prompt on ``pServer``."""

  @pServer.prompt(
    name="edit_podcast",
    title="Edit a podcast end to end",
    description="A full editing workflow: analyse, transcribe, review candidates, plan, simulate, render.",
  )
  @fMcpParameters()
  def fEditPodcastPrompt(
    pFile: str = "Podcast.wav",
    pProfile: str = "podcast-default",
    pGoals: str = "Remove unnecessary silences, obvious mistakes and accidental repetitions. Keep natural pauses.",
  ) -> str:
    vActive = pState.context.profile.name
    return f"""You are editing the podcast recording `{pFile}` with the `{pProfile}` profile
(the server's active profile is `{vActive}`).

Goals stated by the user:
{pGoals}

The quickest road is two calls: `prepare_episode_edit` (the server finds the
retakes, the talk about the recording and the pauses, places the cuts and
checks every join), then `finish_episode_edit` with a `cut` or `keep` decision
for each of its `review_items`, and as `extra_cuts` (seconds of the original)
the stumbles you find reading the transcript that no detector caught: a
fragment said again with other first words, a restarted word, a
self-correction, a muttered aside. If the profile asks to review earlier takes
that say something the later one does not, ask the user about them. With
`mastering=false` the edit is delivered unmastered (`<name>.wav`), and
`master_audio` on that file applies the profile's clean-up and mastering.

When you want to take every decision yourself, work in this order and do not
skip the review step:

1. `get_server_info` - confirm transcription and Audacity are usable.
2. `probe_audio` and `analyze_audio` on `{pFile}` - get the duration, loudness,
   noise floor and silence map. These measurements, not your intuition, are
   what justify each edit.
3. `transcribe_audio` - needed before any word-level decision. On a long file
   this takes a while; it is cached afterwards.
4. `find_retake_candidates`, `find_filler_word_candidates`,
   `find_repetition_candidates` and `find_false_start_candidates`. **Every
   result is a candidate, not a decision.** For each one, read the surrounding
   words with `get_transcript_segment` and decide whether removing it would
   change the meaning or break the rhythm. Reject anything you are unsure
   about. The last complete take of anything said twice is the good one.
5. `create_automatic_edit_plan` - this gives you the silence and mastering
   operations. Add `delete_range` operations for the candidates you accepted,
   using the exact `start` and `end` from the candidate.
6. `validate_edit_plan`, then `simulate_edit_plan` - check the estimated final
   duration and the removed percentage. If more than about a quarter of the
   recording would disappear, stop and ask the user.
7. `execute_edit_plan` - render the final audio.
8. Report the result: original duration, final duration, what was removed and
   why, and the loudness before and after.

Rules:
- Timestamps in a plan are always coordinates of the *original* file. Never
  adjust them for earlier cuts; the engine does that.
- Never delete a region you have not read the transcript for.
- The source file is never modified, so a mistake costs a re-render, not data.
"""

  @pServer.prompt(
    name="review_edit_candidates",
    title="Review edit candidates",
    description="Decide which detected candidates are genuinely worth removing.",
  )
  @fMcpParameters()
  def fReviewCandidatesPrompt(pFile: str = "Podcast.wav") -> str:
    return f"""Review the edit candidates for `{pFile}` and decide which ones to act on.

For each candidate:
- Call `get_transcript_segment` for a window a few seconds wider than the
  candidate, so you can see what comes before and after.
- Ask whether removing it changes the meaning, drops a necessary pause, or
  leaves the sentence ungrammatical.
- Watch for false positives: "esto" and "bueno" are often ordinary words, and a
  repeated phrase is sometimes deliberate emphasis.
- For a repetition, decide which occurrence to keep. Usually the second attempt
  is the complete one, but read both.

Then report a table of accepted and rejected candidates with a one-line reason
for each, and build the `delete_range` operations only for the accepted ones.
"""

  @pServer.prompt(
    name="troubleshoot_podcast_editor_mcp_server",
    title="Troubleshoot the server",
    description="Diagnose why processing, transcription or Audacity is not working.",
  )
  @fMcpParameters()
  def fTroubleshootPrompt(pSymptom: str = "Something is not working") -> str:
    return f"""Diagnose this problem with the podcast-editor-mcp-server server: {pSymptom}

Steps:
1. `health_check` with `check_model=true` - it lists only failing and warning
   checks.
2. If transcription is failing: the model may not be downloaded, or
   faster-whisper may not be installed. The check output names which.
3. If Audacity is unavailable: call `get_audacity_status`. The common cause is
   that `mod-script-pipe` is not enabled. It can be fixed with
   `enable_audacity_scripting`, but **Audacity must be closed first**, because
   it rewrites its configuration file on exit.
4. If a path is rejected: paths must live inside the workspace roots listed by
   `get_server_info`. Copy the file into the input directory instead of
   pointing at it elsewhere.
5. If a plan is rejected: read the validator's `errors` array. Each entry names
   the operation and the rule it broke.

Report what you found and the smallest change that fixes it. Do not suggest
disabling validation.
"""
