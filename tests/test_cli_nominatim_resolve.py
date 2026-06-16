"""Tests for Nominatim probe + text config resolution in ``cli.utils``."""

from __future__ import annotations

import pytest
import typer

from gpxmapper.cli import utils as cli_utils
from gpxmapper.models import TextConfig


def test_resolve_skips_when_not_geolocate() -> None:
    cfg = TextConfig(font_scale=1.0, geolocate=False)
    assert cli_utils._resolve_text_config_after_nominatim_probe(cfg) is cfg


def test_resolve_keeps_geolocate_when_probe_ok(mocker) -> None:
    mocker.patch.object(cli_utils, "probe_nominatim_status_sync", return_value=(True, None))
    cfg = TextConfig(font_scale=1.0, geolocate=True)
    out = cli_utils._resolve_text_config_after_nominatim_probe(cfg)
    assert out.geolocate is True


def test_resolve_drops_geolocate_when_user_confirms(mocker) -> None:
    mocker.patch.object(cli_utils, "probe_nominatim_status_sync", return_value=(False, "connection refused"))
    mocker.patch.object(cli_utils.sys.stdin, "isatty", return_value=True)
    mocker.patch.object(cli_utils.typer, "confirm", return_value=True)
    mocker.patch.object(cli_utils.typer, "secho")
    cfg = TextConfig(font_scale=1.0, geolocate=True)
    out = cli_utils._resolve_text_config_after_nominatim_probe(cfg)
    assert out.geolocate is False


def test_resolve_aborts_when_user_declines(mocker) -> None:
    mocker.patch.object(cli_utils, "probe_nominatim_status_sync", return_value=(False, "down"))
    mocker.patch.object(cli_utils.sys.stdin, "isatty", return_value=True)
    mocker.patch.object(cli_utils.typer, "confirm", return_value=False)
    mocker.patch.object(cli_utils.typer, "secho")
    cfg = TextConfig(font_scale=1.0, geolocate=True)
    with pytest.raises(typer.Abort):
        cli_utils._resolve_text_config_after_nominatim_probe(cfg)


def test_resolve_aborts_without_tty(mocker) -> None:
    mocker.patch.object(cli_utils, "probe_nominatim_status_sync", return_value=(False, "down"))
    mocker.patch.object(cli_utils.sys.stdin, "isatty", return_value=False)
    mocker.patch.object(cli_utils.typer, "secho")
    cfg = TextConfig(font_scale=1.0, geolocate=True)
    with pytest.raises(typer.Abort):
        cli_utils._resolve_text_config_after_nominatim_probe(cfg)
