"""Declarative command registry.

One place defines every command: its flags, types, help, examples and handler.
The argparse tree, the JSON schema returned by ``mne-cli describe`` and the
MCP tool list are all generated from the same table, so an agent can never
disagree with the CLI about what exists.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import UsageError


# --------------------------------------------------------------------- params
@dataclass
class Param:
    """A single CLI option."""

    name: str                      # e.g. "l-freq" -> --l-freq
    type: str = "str"              # str|int|float|bool|json|csv|path|choice
    help: str = ""
    required: bool = False
    default: Any = None
    choices: list[str] | None = None
    metavar: str | None = None
    example: Any = None
    positional: bool = False

    @property
    def flag(self) -> str:
        return f"--{self.name}"

    @property
    def dest(self) -> str:
        return self.name.replace("-", "_")

    def to_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "name": self.name,
            "flag": self.flag,
            "type": self.type,
            "required": self.required,
            "help": self.help,
        }
        if self.default is not None:
            schema["default"] = self.default
        if self.choices:
            schema["choices"] = self.choices
        if self.example is not None:
            schema["example"] = self.example
        return schema


class _BoolAction(argparse.Action):
    """Accept ``--flag``, ``--flag true`` and ``--flag=false`` alike."""

    def __init__(self, option_strings, dest, default=None, **kwargs):
        super().__init__(option_strings, dest, nargs="?", const=True, default=default, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            low = values.strip().lower()
            if low in {"true", "1", "yes", "on", "y"}:
                values = True
            elif low in {"false", "0", "no", "off", "n"}:
                values = False
            else:
                raise argparse.ArgumentError(self, f"expected a boolean, got '{values}'")
        setattr(namespace, self.dest, values)


def _json_arg(raw: Any):
    if raw is None:
        return None
    if isinstance(raw, (dict, list, int, float, bool)):
        return raw
    text = str(raw).strip()
    if text.startswith("@"):
        path = Path(text[1:]).expanduser()
        if not path.exists():
            raise UsageError(f"JSON file not found: {path}", hint="Pass inline JSON or a path to an existing .json file.")
        text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise UsageError(
            f"invalid JSON for this argument: {exc.msg} at position {exc.pos}",
            hint=f"Value was: {raw!r}. Wrap strings in double quotes, e.g. --event-id '{{\"audio\":1}}'.",
        ) from None


def coerce(param: Param, raw: Any):
    """Turn a raw argparse string into the python value a handler expects."""
    if raw is None:
        return None
    if param.type == "bool":
        return bool(raw)
    if param.type == "int":
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise UsageError(f"{param.flag} expects an integer, got {raw!r}") from None
    if param.type == "float":
        if raw in {"inf", "infinity"}:
            return float("inf")
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise UsageError(f"{param.flag} expects a number, got {raw!r}") from None
    if param.type == "json":
        return _json_arg(raw)
    if param.type == "csv":
        if isinstance(raw, list):
            return [str(v) for v in raw]
        return [p.strip() for p in str(raw).split(",") if p.strip()]
    if param.type == "path":
        return str(Path(str(raw)).expanduser())
    return str(raw)


# -------------------------------------------------------------------- command
@dataclass
class Command:
    name: str                                   # "raw.filter"
    summary: str
    handler: Callable[["Ctx", dict[str, Any]], Any]
    params: list[Param] = field(default_factory=list)
    returns: str = "object"
    examples: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    tags: tuple[str, ...] = ()

    @property
    def group(self) -> str:
        return self.name.split(".")[0]

    @property
    def action(self) -> str:
        return self.name.split(".", 1)[1]


REGISTRY: dict[str, Command] = {}

# Destination names the parser itself owns; a command flag may not reuse them.
RESERVED_DEST = {"_group", "_action", "store", "format", "log_level", "json", "text"}


def command(
    name: str,
    summary: str,
    *,
    params: list[Param] | None = None,
    returns: str = "object",
    examples: list[dict[str, Any]] | None = None,
    notes: str = "",
    tags: tuple[str, ...] = (),
):
    """Decorator registering a handler under ``name``."""

    def wrap(fn):
        if name in REGISTRY:
            raise RuntimeError(f"duplicate command {name}")
        for param in params or []:
            if param.dest in RESERVED_DEST:
                raise RuntimeError(f"{name}: parameter '{param.name}' collides with a global option")
        REGISTRY[name] = Command(
            name=name,
            summary=summary,
            handler=fn,
            params=params or [],
            returns=returns,
            examples=examples or [],
            notes=notes,
            tags=tags,
        )
        return fn

    return wrap


def get(name: str) -> Command:
    key = name.replace(" ", ".").strip()
    if key not in REGISTRY:
        close = [n for n in REGISTRY if n.endswith(key) or key in n]
        hint = f" Did you mean: {', '.join(sorted(close)[:5])}?" if close else " Run `mne-cli commands`."
        raise UsageError(f"unknown command: {name}.{hint}")
    return REGISTRY[key]


# ------------------------------------------------------------------ argparse
class _Parser(argparse.ArgumentParser):
    """argparse that reports errors as JSON envelopes instead of dying silently."""

    def error(self, message):  # type: ignore[override]
        raise UsageError(message, hint="Run `mne-cli commands` to list commands, `mne-cli describe <command>` for parameters.")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", default=argparse.SUPPRESS, help="directory holding handles (default ./.mne-cli)")
    common.add_argument("--format", choices=["json", "text"], default=argparse.SUPPRESS, help="output format")
    common.add_argument("--json", dest="format", action="store_const", const="json", default=argparse.SUPPRESS, help="shortcut for --format json")
    common.add_argument("--text", dest="format", action="store_const", const="text", default=argparse.SUPPRESS, help="shortcut for --format text")
    common.add_argument(
        "--log-level",
        default=argparse.SUPPRESS,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="python logging level forwarded to MNE",
    )

    root = _Parser(
        prog="mne-cli",
        description="LLM-friendly command line interface for MNE-Python.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Start with:  mne-cli commands          # discover everything\n"
            "             mne-cli describe raw.filter # exact parameters\n"
            "             mne-cli dataset download sample\n"
        ),
        parents=[common],
    )
    root.set_defaults(store=None, format="json", log_level="WARNING")

    groups: dict[str, argparse.ArgumentParser] = {}
    subs = root.add_subparsers(dest="_group", metavar="<group>", parser_class=_Parser)

    for name in sorted(REGISTRY):
        cmd = REGISTRY[name]
        tokens = name.split(".")
        if len(tokens) == 1:  # top-level verb: mne-cli commands / describe / run
            leaf = subs.add_parser(
                name,
                help=cmd.summary,
                description=cmd.summary,
                parents=[common],
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog=_epilog(cmd),
            )
            leaf.set_defaults(_action=None)
            for param in cmd.params:
                _add_param(leaf, param)
            continue

        group = tokens[0]
        if group not in groups:
            groups[group] = subs.add_parser(
                group,
                help=f"{group} commands",
                parents=[common],
                add_help=True,
            )
            groups[group].add_subparsers(dest="_action", metavar="<action>")
        group_parser = groups[group]
        leaf = group_parser._subparsers._group_actions[0].add_parser(  # type: ignore[attr-defined]
            cmd.action,
            help=cmd.summary,
            description=cmd.summary,
            parents=[common],
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=_epilog(cmd),
        )
        for param in cmd.params:
            _add_param(leaf, param)

    return root


def dispatch_name(ns: argparse.Namespace) -> str | None:
    """Turn a parsed namespace into a registry key."""
    group = getattr(ns, "_group", None)
    if not group:
        return None
    action = getattr(ns, "_action", None)
    return group if not action else f"{group}.{action}"


def _epilog(cmd: Command) -> str:
    lines = []
    if cmd.returns:
        lines.append(f"returns: {cmd.returns}")
    if cmd.notes:
        lines.append(cmd.notes)
    if cmd.examples:
        lines.append("examples:")
        for ex in cmd.examples:
            lines.append(f"  mne-cli {ex.get('args', '')}")
    return "\n".join(lines)


def _add_param(parser: argparse.ArgumentParser, param: Param) -> None:
    if param.positional:
        if param.required:
            parser.add_argument(param.name, metavar=param.metavar or param.name, help=param.help)
        else:
            parser.add_argument(param.name, nargs="?", default=param.default, metavar=param.metavar or param.name, help=param.help)
        return
    kwargs: dict[str, Any] = {"dest": param.dest, "help": param.help, "default": argparse.SUPPRESS}
    if param.metavar:
        kwargs["metavar"] = param.metavar
    if param.type == "bool":
        parser.add_argument(param.flag, action=_BoolAction, default=param.default if param.default is not None else False, **{k: v for k, v in kwargs.items() if k != "default"})
        parser.add_argument(f"--no-{param.name}", dest=param.dest, action="store_false", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        return
    if param.type == "choice":
        kwargs["choices"] = param.choices
    if param.type == "json":
        kwargs["metavar"] = param.metavar or "JSON"
    if param.type == "csv":
        kwargs["metavar"] = param.metavar or "LIST"
    if param.required:
        kwargs["required"] = True
    parser.add_argument(param.flag, **kwargs)


# -------------------------------------------------------------------- schema
def command_schema(cmd: Command) -> dict[str, Any]:
    return {
        "command": cmd.name,
        "summary": cmd.summary,
        "group": cmd.group,
        "params": [p.to_schema() for p in cmd.params],
        "returns": cmd.returns,
        "examples": cmd.examples,
        "notes": cmd.notes,
        "tags": list(cmd.tags),
    }


def list_commands() -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(REGISTRY):
        cmd = REGISTRY[name]
        groups.setdefault(cmd.group, []).append(
            {"command": cmd.name, "summary": cmd.summary, "n_params": len(cmd.params)}
        )
    return {"n_commands": len(REGISTRY), "groups": groups}


JSON_TYPES = {
    "str": "string",
    "path": "string",
    "choice": "string",
    "csv": "array",
    "json": ["object", "array", "string", "number", "boolean"],
    "bool": "boolean",
    "int": "integer",
    "float": "number",
}


def mcp_tools() -> list[dict[str, Any]]:
    tools = []
    for name in sorted(REGISTRY):
        cmd = REGISTRY[name]
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in cmd.params:
            prop: dict[str, Any] = {"description": p.help, "type": JSON_TYPES.get(p.type, "string")}
            if p.default is not None:
                prop["default"] = p.default
            if p.choices:
                prop["enum"] = p.choices
            props[p.dest] = prop
            if p.required:
                required.append(p.dest)
        tools.append(
            {
                "name": f"mne_cli_{cmd.name.replace('.', '_')}",
                "description": f"[{cmd.name}] {cmd.summary} Returns: {cmd.returns}",
                "inputSchema": {"type": "object", "properties": props, "required": required, "additionalProperties": False},
                "_command": cmd.name,
            }
        )
    return tools
