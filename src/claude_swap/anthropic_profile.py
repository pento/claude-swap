"""Anthropic-profile slots: a slot that stands for a Claude Code Console login.

A keyless Console sign-in (``/login``, then "Anthropic Console account", then
signing in without creating an API key) does not write Claude Code's
``.credentials.json`` or its Keychain item. It writes an *Anthropic profile*
instead (read from Claude Code 2.1.332):

- The config dir is ``$ANTHROPIC_CONFIG_DIR``, else
  ``$XDG_CONFIG_HOME/anthropic``, else ``~/.config/anthropic``
  (``paths.get_anthropic_config_dir``).
- ``configs/<name>.json`` holds the organization, workspace and OAuth client.
- ``credentials/<name>.json`` holds the OAuth tokens, ``expires_at`` in
  seconds, and the ``organization_uuid`` and ``organization_name`` they belong
  to.
- ``active_config`` names the profile used when ``ANTHROPIC_PROFILE`` is
  unset; with neither, the name is ``default``. A name is letters, digits,
  ``_``, ``.`` and ``-``.
- With a claude.ai login present Claude Code skips the implicit profile. With
  ``ANTHROPIC_PROFILE`` set it uses that profile and not a claude.ai login.

A profile slot is a ``sequence.json`` record with ``kind: "profile"`` and
``profile: <name>``, and nothing else: no credential and no config backup.
Claude Code refreshes the profile's token itself, so cswap never copies,
refreshes or stores it, and reads only the organization fields and
``expires_at`` from the credential file, never a token.

Such a slot is for ``cswap run`` only. ``SessionManager`` launches it in the
slot's session profile with ``CLAUDE_CONFIG_DIR`` and ``ANTHROPIC_PROFILE``
set, so the default login is never involved. It is never a switch target,
never in rotation, and has no usage to fetch.

This module is a leaf: it reads files and environment, and imports nothing
from the package beyond ``paths``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from claude_swap.paths import get_anthropic_config_dir

#: The ``kind`` of a profile slot's ``sequence.json`` record.
PROFILE_KIND = "profile"

#: The domain of a profile slot's default label, ``<name>@<domain>``.
LABEL_DOMAIN = "anthropic-profile.local"

#: Claude Code's profile-name rule.
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_profile_name(name: str) -> str:
    """Return ``name`` if Claude Code would accept it as a profile name.

    Names made only of dots are refused as well: the name becomes a file
    name under the config dir, and ``.`` or ``..`` would leave it.

    Raises:
        ValueError: the name is empty or breaks the rule.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name) or set(name) == {"."}:
        raise ValueError(
            f"Invalid Anthropic profile name {name!r}: use letters, digits, "
            "'_', '.' and '-'."
        )
    return name


def default_label(name: str) -> str:
    """The slot label used when ``cswap add --profile`` is given no ``--email``."""
    return f"{name}@{LABEL_DOMAIN}"


def credentials_path(name: str) -> Path:
    """Where Claude Code keeps the profile's OAuth credential."""
    return get_anthropic_config_dir() / "credentials" / f"{validate_profile_name(name)}.json"


def active_profile_name() -> str:
    """The profile Claude Code would use in this environment.

    ``ANTHROPIC_PROFILE``, else the first line of ``active_config``, else
    ``default``. An unreadable or invalid ``active_config`` reads as
    ``default``.
    """
    env = os.environ.get("ANTHROPIC_PROFILE")
    if env:
        return env
    try:
        text = (get_anthropic_config_dir() / "active_config").read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError):
        return "default"
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    try:
        return validate_profile_name(first)
    except ValueError:
        return "default"


@dataclass(frozen=True)
class ProfileState:
    """What cswap may know about a profile's login. Never holds a token."""

    signed_in: bool
    organization_uuid: str = ""
    organization_name: str = ""
    #: The access token's expiry (epoch seconds), when recorded. Claude Code
    #: refreshes past it while the refresh token is good.
    expires_at: float | None = None


def read_profile_state(name: str) -> ProfileState:
    """Read the profile's credential file for its login state.

    Signed in means the file is a JSON object holding a non-empty access or
    refresh token. Only the organization fields and ``expires_at`` are
    returned. An invalid name, a missing or unreadable file, and a file of
    another shape all read as not signed in.
    """
    try:
        raw = credentials_path(name).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, ValueError):
        return ProfileState(signed_in=False)
    if not isinstance(data, dict):
        return ProfileState(signed_in=False)
    signed_in = any(
        isinstance(data.get(key), str) and data.get(key)
        for key in ("access_token", "refresh_token")
    )
    expires_at = data.get("expires_at")
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        expires_at = None
    org_uuid = data.get("organization_uuid")
    org_name = data.get("organization_name")
    return ProfileState(
        signed_in=signed_in,
        organization_uuid=org_uuid if isinstance(org_uuid, str) else "",
        organization_name=org_name if isinstance(org_name, str) else "",
        expires_at=float(expires_at) if expires_at is not None else None,
    )


def profile_name_of(record: dict | None) -> str | None:
    """The profile a ``sequence.json`` record stands for, or ``None``.

    ``None`` for every record that is not a profile slot, and for a profile
    record whose name is missing or invalid (it then launches nothing).
    """
    if not isinstance(record, dict) or record.get("kind") != PROFILE_KIND:
        return None
    try:
        return validate_profile_name(record.get("profile", ""))
    except ValueError:
        return None


def is_profile_record(record: dict | None) -> bool:
    """Whether a ``sequence.json`` record is a profile slot, valid or not."""
    return isinstance(record, dict) and record.get("kind") == PROFILE_KIND


def not_signed_in_notice(name: str) -> str:
    """What to do in a session whose profile has no login yet."""
    return (
        f"Anthropic profile '{name}' is not signed in. In this session run "
        "/login, choose \"Anthropic Console account\" and sign in with your "
        "Console account. That replaces only this session profile's sign-in, "
        "not your default login."
    )


def run_only_message(account_num: str, name: str) -> str:
    """The refusal for any attempt to switch to a profile slot."""
    return (
        f"Account-{account_num} is Anthropic profile '{name}'; profile slots "
        f"are for `cswap run` only. Start it with: cswap run {account_num}"
    )
