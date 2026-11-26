"""Tests for Anthropic-profile slots (``claude_swap.anthropic_profile``).

Covers the module's own reads (config dir, name rule, active profile, login
state without tokens), ``cswap add --profile``, the ``cswap run`` launch
environment, the switch and rotation refusals, ``list`` and ``--json``
output, the ``cswap add`` guard against a capture with no login, export and
import of the record, and that ``remove`` leaves the Anthropic config dir
alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_swap import anthropic_profile
from claude_swap import session as session_mod
from claude_swap.cli import main
from claude_swap.exceptions import (
    ConfigError,
    SessionError,
    SwitchError,
    ValidationError,
)
from claude_swap.json_output import USAGE_ANTHROPIC_PROFILE, usage_fields
from claude_swap.models import Platform
from claude_swap.paths import get_anthropic_config_dir
from claude_swap.session import SessionManager, session_dir_for
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.transfer import export_accounts, import_accounts

ACCESS = "sk-ant-oat01-profile-access-SECRET"
REFRESH = "sk-ant-ort01-profile-refresh-SECRET"
OAUTH_CREDS = json.dumps(
    {"claudeAiOauth": {"accessToken": "tok", "refreshToken": "rtok", "expiresAt": 9}}
)


def _write_profile(name: str = "mythos", **overrides) -> Path:
    """Write ``credentials/<name>.json`` the way Claude Code 2.1.332 does."""
    data = {
        "type": "oauth_token",
        "access_token": ACCESS,
        "refresh_token": REFRESH,
        "expires_at": 4102444800,  # 2100-01-01
        "scope": "user:inference",
        "organization_uuid": "org-console",
        "organization_name": "Console Org",
        "account_email": "person@example.com",
        "workspace_id": "wrk",
        "workspace_name": "Mythos",
    }
    data.update(overrides)
    path = get_anthropic_config_dir() / "credentials" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def switcher(temp_home: Path, monkeypatch) -> ClaudeAccountSwitcher:
    monkeypatch.setattr(Platform, "detect", classmethod(lambda cls: Platform.LINUX))
    s = ClaudeAccountSwitcher()
    s.platform = Platform.LINUX
    s._setup_directories()
    s._init_sequence_file()
    return s


def _seed_oauth_slot(s: ClaudeAccountSwitcher, num: str, email: str) -> None:
    data = s._get_sequence_data()
    data["accounts"][num] = {
        "email": email,
        "uuid": f"uuid-{num}",
        "organizationUuid": "",
        "organizationName": "",
        "added": "2024-01-01T00:00:00Z",
    }
    data["sequence"] = sorted({*data["sequence"], int(num)})
    s._write_json(s.sequence_file, data)
    s._write_account_credentials(num, email, OAUTH_CREDS)
    s._write_account_config(
        num, email, json.dumps({"oauthAccount": {"emailAddress": email}})
    )


# ---------------------------------------------------------------------------
# The module's own reads
# ---------------------------------------------------------------------------


class TestConfigDir:
    def test_default_is_dot_config_anthropic(self, temp_home: Path):
        assert get_anthropic_config_dir() == temp_home / ".config" / "anthropic"

    def test_xdg_config_home(self, temp_home: Path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(temp_home / "xdg"))
        assert get_anthropic_config_dir() == temp_home / "xdg" / "anthropic"

    def test_anthropic_config_dir_wins(self, temp_home: Path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(temp_home / "xdg"))
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(temp_home / "explicit"))
        assert get_anthropic_config_dir() == temp_home / "explicit"


class TestProfileName:
    @pytest.mark.parametrize("name", ["mythos", "a.b-c_d", "Default9"])
    def test_valid(self, name):
        assert anthropic_profile.validate_profile_name(name) == name

    @pytest.mark.parametrize(
        "name", ["", "a/b", "a b", "..", ".", "../x", "naïve", "a\\b"]
    )
    def test_invalid(self, name):
        with pytest.raises(ValueError):
            anthropic_profile.validate_profile_name(name)

    def test_active_profile_env_wins(self, temp_home: Path, monkeypatch):
        (get_anthropic_config_dir()).mkdir(parents=True)
        (get_anthropic_config_dir() / "active_config").write_text("work\n")
        monkeypatch.setenv("ANTHROPIC_PROFILE", "mythos")
        assert anthropic_profile.active_profile_name() == "mythos"

    def test_active_profile_from_file(self, temp_home: Path):
        get_anthropic_config_dir().mkdir(parents=True)
        (get_anthropic_config_dir() / "active_config").write_text("work\n")
        assert anthropic_profile.active_profile_name() == "work"

    def test_active_profile_default(self, temp_home: Path):
        assert anthropic_profile.active_profile_name() == "default"

    def test_invalid_active_config_reads_default(self, temp_home: Path):
        get_anthropic_config_dir().mkdir(parents=True)
        (get_anthropic_config_dir() / "active_config").write_text("../evil\n")
        assert anthropic_profile.active_profile_name() == "default"


class TestProfileState:
    def test_signed_in_carries_org_and_expiry_but_no_token(self, temp_home: Path):
        _write_profile()
        state = anthropic_profile.read_profile_state("mythos")
        assert state.signed_in is True
        assert state.organization_uuid == "org-console"
        assert state.organization_name == "Console Org"
        assert state.expires_at == 4102444800.0
        assert ACCESS not in repr(state) and REFRESH not in repr(state)

    def test_missing_file_is_not_signed_in(self, temp_home: Path):
        assert anthropic_profile.read_profile_state("mythos").signed_in is False

    def test_empty_tokens_are_not_signed_in(self, temp_home: Path):
        _write_profile(access_token="", refresh_token="")
        assert anthropic_profile.read_profile_state("mythos").signed_in is False

    @pytest.mark.parametrize("text", ["not json", "[]", '"x"'])
    def test_other_shapes_are_not_signed_in(self, temp_home: Path, text):
        path = get_anthropic_config_dir() / "credentials" / "mythos.json"
        path.parent.mkdir(parents=True)
        path.write_text(text)
        assert anthropic_profile.read_profile_state("mythos").signed_in is False

    def test_invalid_name_is_not_signed_in(self, temp_home: Path):
        assert anthropic_profile.read_profile_state("../x").signed_in is False


# ---------------------------------------------------------------------------
# cswap add --profile
# ---------------------------------------------------------------------------


class TestAddProfile:
    def test_registers_record_without_credentials(self, switcher, capsys):
        _write_profile()
        switcher.add_profile_account("mythos", slot=6, alias="myth")

        record = switcher._get_sequence_data()["accounts"]["6"]
        assert record["kind"] == "profile"
        assert record["profile"] == "mythos"
        assert record["email"] == "mythos@anthropic-profile.local"
        assert record["organizationUuid"] == "org-console"
        assert record["organizationName"] == "Console Org"
        assert record["alias"] == "myth"
        assert 6 in switcher._get_sequence_data()["sequence"]
        # Nothing secret is stored for the slot.
        assert switcher._read_account_credentials("6", record["email"]) in ("", None)
        assert switcher._read_account_config("6", record["email"]) == ""
        assert ACCESS not in switcher.sequence_file.read_text()
        out = capsys.readouterr().out
        assert "cswap run only" in out
        assert ACCESS not in out and REFRESH not in out

    def test_not_signed_in_yet_is_allowed(self, switcher, capsys):
        switcher.add_profile_account("mythos", slot=6)
        record = switcher._get_sequence_data()["accounts"]["6"]
        assert record["organizationUuid"] == ""
        assert "Not signed in yet" in capsys.readouterr().out

    def test_email_overrides_label(self, switcher):
        switcher.add_profile_account("mythos", email="me@example.com")
        accounts = switcher._get_sequence_data()["accounts"]
        assert [r["email"] for r in accounts.values()] == ["me@example.com"]

    def test_invalid_name_is_refused(self, switcher):
        with pytest.raises(ValidationError, match="Invalid Anthropic profile"):
            switcher.add_profile_account("../evil")
        assert switcher._get_sequence_data()["accounts"] == {}

    def test_name_already_registered_elsewhere_is_refused(self, switcher):
        switcher.add_profile_account("mythos", slot=6)
        with pytest.raises(ValidationError, match="already Account-6"):
            switcher.add_profile_account("mythos", slot=7)
        with pytest.raises(ValidationError, match="already Account-6"):
            switcher.add_profile_account("mythos")

    def test_readd_to_own_slot_keeps_settings(self, switcher):
        switcher.add_profile_account("mythos", slot=6, alias="myth")
        switcher.set_account_disabled("6", True)
        _write_profile()
        switcher.add_profile_account("mythos", slot=6)
        record = switcher._get_sequence_data()["accounts"]["6"]
        assert record["alias"] == "myth"
        assert record["disabled"] is True
        assert record["organizationUuid"] == "org-console"

    def test_occupied_slot_prompts_and_displaces(self, switcher, monkeypatch):
        _seed_oauth_slot(switcher, "6", "old@example.com")
        monkeypatch.setattr("builtins.input", lambda _prompt: "y")
        switcher.add_profile_account("mythos", slot=6)
        record = switcher._get_sequence_data()["accounts"]["6"]
        assert record["kind"] == "profile"
        assert not switcher._read_account_credentials("6", "old@example.com")

    def test_occupied_slot_declined_changes_nothing(self, switcher, monkeypatch):
        _seed_oauth_slot(switcher, "6", "old@example.com")
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        switcher.add_profile_account("mythos", slot=6)
        assert switcher._get_sequence_data()["accounts"]["6"]["email"] == "old@example.com"

    def test_label_colliding_with_another_slot_is_refused(self, switcher):
        _seed_oauth_slot(switcher, "1", "me@example.com")
        with pytest.raises(ValidationError, match="distinct --email"):
            switcher.add_profile_account("mythos", slot=6, email="me@example.com")

    def test_cli_add_profile(self, switcher, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["cswap", "add", "--profile", "mythos", "--slot", "6", "--alias", "myth"],
        )
        main()
        record = ClaudeAccountSwitcher()._get_sequence_data()["accounts"]["6"]
        assert (record["kind"], record["profile"], record["alias"]) == (
            "profile", "mythos", "myth",
        )

    def test_cli_profile_needs_add(self, temp_home: Path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["cswap", "list", "--profile", "mythos"])
        with pytest.raises(SystemExit):
            main()

    def test_add_token_cannot_overwrite_a_profile_slot_by_email(self, switcher):
        switcher.add_profile_account("mythos", slot=6, email="me@example.com")
        with pytest.raises(ValidationError, match="Anthropic-profile"):
            switcher.add_account_from_token(
                "sk-ant-oat01-" + "x" * 40, email="me@example.com"
            )


# ---------------------------------------------------------------------------
# cswap run
# ---------------------------------------------------------------------------


class _ExecCalled(Exception):
    def __init__(self, argv, env):
        self.argv, self.env = argv, env


@pytest.fixture
def capture_exec(monkeypatch):
    def fake_exec(self, claude_bin, claude_args, env):
        raise _ExecCalled([claude_bin, *claude_args], env)

    monkeypatch.setattr(session_mod.SessionManager, "_exec", fake_exec)
    monkeypatch.setattr(session_mod.shutil, "which", lambda name: f"/fake/bin/{name}")


class TestRun:
    def test_env_names_profile_and_session_dir(
        self, switcher, capture_exec, monkeypatch, capsys
    ):
        _write_profile()
        switcher.add_profile_account("mythos", slot=6)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-should-be-scrubbed")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "should-be-scrubbed")
        monkeypatch.setenv("ANTHROPIC_PROFILE", "other")

        with pytest.raises(_ExecCalled) as exc:
            SessionManager(switcher).run("6", ["--resume"])

        env = exc.value.env
        session_dir = session_dir_for(
            switcher.backup_dir, "6", "mythos@anthropic-profile.local"
        )
        assert env["CLAUDE_CONFIG_DIR"] == str(session_dir)
        assert env["ANTHROPIC_PROFILE"] == "mythos"
        assert "ANTHROPIC_API_KEY" not in env
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
        assert exc.value.argv == ["/fake/bin/claude", "--resume"]
        # No credential is seeded; only onboarding keys.
        assert not (session_dir / ".credentials.json").exists()
        seeded = json.loads((session_dir / ".claude.json").read_text())
        assert seeded["hasCompletedOnboarding"] is True
        assert seeded["theme"]
        assert "oauthAccount" not in seeded
        captured = capsys.readouterr()
        assert "not signed in" not in captured.err
        assert ACCESS not in captured.out + captured.err

    def test_seed_keeps_existing_keys(self, switcher, capture_exec):
        switcher.add_profile_account("mythos", slot=6)
        session_dir = session_dir_for(
            switcher.backup_dir, "6", "mythos@anthropic-profile.local"
        )
        session_dir.mkdir(parents=True)
        (session_dir / ".claude.json").write_text(
            json.dumps({"theme": "light", "projects": {"x": 1}})
        )
        with pytest.raises(_ExecCalled):
            SessionManager(switcher).run("6", [])
        seeded = json.loads((session_dir / ".claude.json").read_text())
        assert seeded == {
            "theme": "light", "projects": {"x": 1}, "hasCompletedOnboarding": True,
        }

    def test_not_signed_in_notice(self, switcher, capture_exec, capsys):
        switcher.add_profile_account("mythos", slot=6)
        with pytest.raises(_ExecCalled):
            SessionManager(switcher).run("6", [])
        out = capsys.readouterr()
        text = out.out + out.err
        assert "not signed in" in text
        assert "/login" in text and "Anthropic Console account" in text

    def test_no_fast_path_seed_or_refresh(self, switcher, capture_exec, monkeypatch):
        _write_profile()
        switcher.add_profile_account("mythos", slot=6)
        # Even when the profile slot's identity looks like the default login,
        # it still launches in its session profile.
        monkeypatch.setattr(
            switcher, "_get_current_account",
            lambda: ("mythos@anthropic-profile.local", "org-console"),
        )

        def boom(*_a, **_k):
            raise AssertionError("must not seed or refresh a profile slot")

        monkeypatch.setattr(SessionManager, "setup_session", boom)
        monkeypatch.setattr(switcher, "consume_backup_grant", boom)
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(switcher).run("6", [])
        assert exc.value.env["ANTHROPIC_PROFILE"] == "mythos"
        assert "CLAUDE_CONFIG_DIR" in exc.value.env

    def test_invalid_stored_name_is_refused(self, switcher, capture_exec):
        switcher.add_profile_account("mythos", slot=6)
        data = switcher._get_sequence_data()
        data["accounts"]["6"]["profile"] = "../x"
        switcher._write_json(switcher.sequence_file, data)
        with pytest.raises(SessionError, match="no valid profile name"):
            SessionManager(switcher).run("6", [])

    def test_oauth_slot_is_untouched_by_the_hook(self, switcher, capture_exec):
        _seed_oauth_slot(switcher, "2", "two@example.com")
        hook = SessionManager._run_if_anthropic_profile
        manager = SessionManager(switcher)
        # Returns (no exec) for an ordinary slot.
        assert hook(manager, "2", "/fake/bin/claude", [], True, True) is None


# ---------------------------------------------------------------------------
# Kept out of switch, rotation and usage
# ---------------------------------------------------------------------------


class TestSwitchRefusal:
    def test_switch_to_refuses(self, switcher):
        switcher.add_profile_account("mythos", slot=6)
        with pytest.raises(SwitchError, match="cswap run"):
            switcher.switch_to("6")

    def test_perform_switch_refuses(self, switcher):
        switcher.add_profile_account("mythos", slot=6)
        with pytest.raises(SwitchError, match="cswap run"):
            switcher._perform_switch("6")

    def test_not_switchable_and_not_in_rotation(self, switcher):
        _seed_oauth_slot(switcher, "1", "one@example.com")
        switcher.add_profile_account("mythos", slot=6)
        assert switcher._account_is_switchable("6") is False
        assert switcher.switchable_account_numbers() == ["1"]
        assert switcher.account_kind_for("6") == "profile"

    def test_rotation_skips_with_reason(self, switcher, monkeypatch, capsys):
        _seed_oauth_slot(switcher, "1", "one@example.com")
        switcher.add_profile_account("mythos", slot=2)
        _seed_oauth_slot(switcher, "3", "three@example.com")
        monkeypatch.setattr(
            switcher, "_get_current_account", lambda: ("one@example.com", "")
        )
        targets = []
        monkeypatch.setattr(
            switcher, "_perform_switch",
            lambda target, **_k: targets.append(target) or {
                "from": None, "to": None, "warnings": [],
            },
        )
        switcher.switch()
        assert targets == ["3"]
        assert "Anthropic profile, cswap run only" in capsys.readouterr().out


class TestUsageAndList:
    def test_static_sentinel(self, switcher):
        switcher.add_profile_account("mythos", slot=6)
        info = (6, "mythos@anthropic-profile.local", "", "", False, "", "")
        assert switcher._static_usage_sentinel(info) == USAGE_ANTHROPIC_PROFILE
        assert usage_fields(USAGE_ANTHROPIC_PROFILE) == ("anthropic_profile", None)

    def test_never_fetched(self, switcher, monkeypatch):
        switcher.add_profile_account("mythos", slot=6)

        def boom(*_a, **_k):
            raise AssertionError("profile slots have no usage to fetch")

        monkeypatch.setattr(switcher, "_fetch_account_usage", boom)
        monkeypatch.setattr(switcher, "_fetch_active_usage", boom)
        payload = switcher.list_accounts(json_output=True)
        assert payload["accounts"][0]["usageStatus"] == "anthropic_profile"

    def test_json_row_fields(self, switcher):
        _write_profile()
        switcher.add_profile_account("mythos", slot=6)
        row = switcher.list_accounts(json_output=True)["accounts"][0]
        assert row["kind"] == "profile"
        assert row["profile"] == "mythos"
        assert row["profileSignedIn"] is True
        assert row["profileTokenExpiresAt"] == "2100-01-01T00:00:00Z"
        assert row["active"] is False
        assert row["usage"] is None
        assert ACCESS not in json.dumps(row) and REFRESH not in json.dumps(row)

    def test_json_rows_of_other_slots_unchanged(self, switcher):
        _seed_oauth_slot(switcher, "1", "one@example.com")
        row = switcher.list_accounts(json_output=True)["accounts"][0]
        assert not {"kind", "profile", "profileSignedIn"} & set(row)

    def test_human_list(self, switcher, capsys):
        _write_profile()
        switcher.add_profile_account("mythos", slot=6)
        capsys.readouterr()
        switcher.list_accounts(show_token_status=True)
        out = capsys.readouterr().out
        assert "Anthropic profile mythos · cswap run only" in out
        assert "signed in · token expires" in out
        assert ACCESS not in out and REFRESH not in out

    def test_human_list_not_signed_in(self, switcher, capsys):
        switcher.add_profile_account("mythos", slot=6)
        capsys.readouterr()
        switcher.list_accounts()
        assert "not signed in · run: cswap run 6, then /login" in capsys.readouterr().out

    def test_snapshot_kind(self, switcher):
        switcher.add_profile_account("mythos", slot=6)
        snapshot = switcher.accounts_snapshot()
        (acc,) = snapshot.accounts
        assert acc.kind == "profile"
        assert acc.switchable is False


# ---------------------------------------------------------------------------
# cswap add guard
# ---------------------------------------------------------------------------


class TestAddGuard:
    def _live_identity(self, temp_home: Path) -> None:
        (temp_home / ".claude.json").write_text(json.dumps({
            "oauthAccount": {
                "emailAddress": "person@example.com",
                "accountUuid": "uuid-p",
                "organizationUuid": "org-p",
            }
        }))

    def test_mcp_only_capture_is_refused_with_profile_hint(
        self, switcher, temp_home: Path, monkeypatch
    ):
        self._live_identity(temp_home)
        _write_profile("mythos")
        get_anthropic_config_dir().joinpath("active_config").write_text("mythos\n")
        monkeypatch.setattr(
            switcher, "_read_capture_credentials",
            lambda: json.dumps({"mcpOAuth": {"server": {"accessToken": "m"}}}),
        )
        with pytest.raises(ConfigError, match="cswap add --profile mythos"):
            switcher.add_account(slot=6, assume_yes=True)
        assert switcher._get_sequence_data()["accounts"] == {}

    def test_mcp_only_capture_without_profile(
        self, switcher, temp_home: Path, monkeypatch
    ):
        self._live_identity(temp_home)
        monkeypatch.setattr(
            switcher, "_read_capture_credentials",
            lambda: json.dumps({"mcpOAuth": {}}),
        )
        with pytest.raises(ConfigError, match="no Claude Code OAuth tokens") as exc:
            switcher.add_account()
        assert "--profile" not in str(exc.value)
        assert switcher._get_sequence_data()["accounts"] == {}

    def test_legacy_top_level_tokens_still_register(
        self, switcher, temp_home: Path, monkeypatch
    ):
        self._live_identity(temp_home)
        monkeypatch.setattr(
            switcher, "_read_capture_credentials",
            lambda: json.dumps({"accessToken": "t", "refreshToken": "r"}),
        )
        switcher.add_account(slot=3, assume_yes=True)
        assert "3" in switcher._get_sequence_data()["accounts"]

    def test_refresh_in_place_refuses_a_profile_slot(
        self, switcher, temp_home: Path, monkeypatch
    ):
        self._live_identity(temp_home)
        _write_profile(organization_uuid="org-p")
        switcher.add_profile_account("mythos", slot=6, email="person@example.com")
        monkeypatch.setattr(
            switcher, "_read_capture_credentials", lambda: OAUTH_CREDS
        )
        with pytest.raises(ValidationError, match="Anthropic-profile slot"):
            switcher.add_account()
        assert switcher._get_sequence_data()["accounts"]["6"]["kind"] == "profile"


# ---------------------------------------------------------------------------
# export / import / remove
# ---------------------------------------------------------------------------


class TestTransferAndRemove:
    def test_export_import_round_trip(self, switcher, temp_home: Path, capsys):
        _write_profile()
        switcher.add_profile_account("mythos", slot=6, alias="myth")
        out = temp_home / "export.json"
        export_accounts(switcher, str(out))
        envelope = json.loads(out.read_text())
        (entry,) = envelope["accounts"]
        assert entry["kind"] == "profile" and entry["profile"] == "mythos"
        assert "credentials" not in entry and "config" not in entry
        assert ACCESS not in out.read_text()

        data = switcher._get_sequence_data()
        del data["accounts"]["6"]
        data["sequence"] = []
        switcher._write_json(switcher.sequence_file, data)

        import_accounts(switcher, str(out))
        record = switcher._get_sequence_data()["accounts"]["6"]
        assert record["kind"] == "profile"
        assert record["profile"] == "mythos"
        assert record["alias"] == "myth"
        assert not switcher._read_account_credentials("6", record["email"])

    def test_import_rejects_invalid_profile_name(self, switcher, temp_home: Path):
        src = temp_home / "bad.json"
        src.write_text(json.dumps({
            "version": 1, "accounts": [{
                "number": 6, "email": "x@anthropic-profile.local",
                "kind": "profile", "profile": "../evil",
            }],
        }))
        from claude_swap.exceptions import TransferError

        with pytest.raises(TransferError, match="profile"):
            import_accounts(switcher, str(src))

    def test_remove_leaves_anthropic_config_alone(self, switcher, capture_exec):
        cred = _write_profile()
        switcher.add_profile_account("mythos", slot=6)
        with pytest.raises(_ExecCalled):
            SessionManager(switcher).run("6", [])
        session_dir = session_dir_for(
            switcher.backup_dir, "6", "mythos@anthropic-profile.local"
        )
        assert session_dir.exists()

        switcher.remove_account("6", assume_yes=True)

        assert "6" not in switcher._get_sequence_data()["accounts"]
        assert not session_dir.exists()
        assert cred.exists()
        assert json.loads(cred.read_text())["access_token"] == ACCESS


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------


class TestProfileEnvDoesNotLeak:
    def test_probe_env_drops_anthropic_profile(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_PROFILE", "mythos")
        assert "ANTHROPIC_PROFILE" not in session_mod._probe_env(tmp_path)

    def test_oauth_session_launch_drops_anthropic_profile(
        self, switcher, capture_exec, monkeypatch
    ):
        """A `cswap run N` from inside a profile session inherits
        ANTHROPIC_PROFILE; the OAuth session must not run on the profile."""
        _seed_oauth_slot(switcher, "1", "me@example.com")
        session_dir = session_dir_for(switcher.backup_dir, "1", "me@example.com")
        monkeypatch.setattr(
            SessionManager,
            "setup_session",
            lambda self, identifier, share, share_history=None: (
                session_dir, "1", "me@example.com"
            ),
        )
        monkeypatch.setattr(ClaudeAccountSwitcher, "_get_current_account", lambda self: None)
        monkeypatch.setenv("ANTHROPIC_PROFILE", "mythos")
        with pytest.raises(_ExecCalled) as exc:
            SessionManager(switcher).run("1", [])
        assert "ANTHROPIC_PROFILE" not in exc.value.env


class TestReviewFixes:
    def test_seed_leaves_unreadable_config_alone(self, switcher, capture_exec):
        switcher.add_profile_account("mythos", slot=6)
        session_dir = session_dir_for(
            switcher.backup_dir, "6", "mythos@anthropic-profile.local"
        )
        session_dir.mkdir(parents=True)
        (session_dir / ".claude.json").write_text('{"projects": {"x": 1}')
        with pytest.raises(_ExecCalled):
            SessionManager(switcher).run("6", [])
        assert (session_dir / ".claude.json").read_text() == '{"projects": {"x": 1}'

    @pytest.mark.parametrize("value", [float("nan"), 1e20, -1e20])
    def test_unrepresentable_expiry_reads_as_unrecorded(
        self, switcher, capsys, value
    ):
        _write_profile(expires_at=value)
        assert anthropic_profile.read_profile_state("mythos").expires_at is None
        switcher.add_profile_account("mythos", slot=6)
        row = switcher._profile_row_fields("6")
        assert "profileTokenExpiresAt" not in row
        switcher.list_accounts()
        assert "signed in" in capsys.readouterr().out

    def test_readd_keeps_label_and_mappings(self, switcher):
        from claude_swap.mappings import MappingStore

        switcher.add_profile_account("mythos", slot=6, email="me@mythos.example")
        store = MappingStore(switcher.backup_dir)
        store.set("/work/a", "me@mythos.example", "")
        _write_profile()  # first sign-in records the organization
        switcher.add_profile_account("mythos", slot=6)
        record = switcher._get_sequence_data()["accounts"]["6"]
        assert record["email"] == "me@mythos.example"
        assert record["organizationUuid"] == "org-console"
        entry = store.get("/work/a")
        assert entry["email"] == "me@mythos.example"
        assert entry["organizationUuid"] == "org-console"

    def test_import_skips_a_profile_already_in_another_slot(
        self, switcher, temp_home: Path, capsys
    ):
        switcher.add_profile_account("mythos", slot=6)
        src = temp_home / "dup.json"
        src.write_text(json.dumps({
            "version": 1, "accounts": [{
                "number": 7, "email": "other@anthropic-profile.local",
                "kind": "profile", "profile": "mythos",
            }],
        }))
        import_accounts(switcher, str(src))
        assert "7" not in switcher._get_sequence_data()["accounts"]
        assert "already slot 6" in capsys.readouterr().err
