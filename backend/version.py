"""Product identity and version.

Kept in a single module so that reports, MCP server info and the CLI all quote
exactly the same string. There is no ``__init__.py`` anywhere in this package:
the project targets Python 3.13+ and relies on implicit namespace packages.
"""

from __future__ import annotations

cProductName: str = "podcast-editor-mcp-server"
cProductTitle: str = "Podcast Editor MCP Server"
cVersion: str = "0.1.0"

# Schema version of the EditPlan document. Bumped independently of cVersion
# whenever the on-disk plan representation changes in a non-additive way.
cEditPlanSchemaVersion: int = 1

# Schema version of the persisted Job document.
cJobSchemaVersion: int = 1
