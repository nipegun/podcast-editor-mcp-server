"""Keep MCP parameter names stable while using project identifiers internally."""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Callable
from typing import Any


def fPublicParameterName(pName: str) -> str:
  """Translate an internal ``pCamelCase`` parameter to its MCP spelling."""
  return re.sub(r"(?<!^)(?=[A-Z])", "_", pName[1:]).lower()


def fMcpParameters(pPublicName: str | None = None) -> Callable:
  """Expose the original signature and map validated MCP keyword arguments.

  The synchronous MCP tools, resources and prompts keep their public names,
  annotations and defaults. Only the identifiers in their implementations
  use the project's naming convention.
  """

  def fDecorate[ReturnT](pFunction: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    vSignature = inspect.signature(pFunction, eval_str=True)
    dInternalNames = {fPublicParameterName(vName): vName for vName in vSignature.parameters}
    lParameters = [
      vParameter.replace(name=fPublicParameterName(vName))
      for vName, vParameter in vSignature.parameters.items()
    ]

    @functools.wraps(pFunction)
    def fWrapper(*pArgs: Any, **pKeywordArgs: Any) -> ReturnT:
      dArguments = {dInternalNames.get(vName, vName): vValue for vName, vValue in pKeywordArgs.items()}
      return pFunction(*pArgs, **dArguments)

    setattr(fWrapper, "__signature__", vSignature.replace(parameters=lParameters))
    fWrapper.__annotations__ = {
      vParameter.name: vParameter.annotation
      for vParameter in lParameters
      if vParameter.annotation is not inspect.Parameter.empty
    }
    if vSignature.return_annotation is not inspect.Signature.empty:
      fWrapper.__annotations__["return"] = vSignature.return_annotation
    if pPublicName is not None:
      fWrapper.__name__ = pPublicName
    return fWrapper

  return fDecorate
