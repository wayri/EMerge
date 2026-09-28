"""Regression checks for stackup-unit inverse solvers and line dispersion.

The calculator uses Material only as a type annotation in these tests, so the
lightweight stub avoids importing EMerge's optional meshing/runtime stack.
"""

import importlib.util
import itertools
import math
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def calc():
    emsutil = types.ModuleType("emsutil")
    emsutil.Material = type("Material", (), {})
    original = sys.modules.get("emsutil")
    sys.modules["emsutil"] = emsutil
    try:
        path = Path(__file__).resolve().parents[1] / "src/emerge/_emerge/geo/pcb_tools/calculator.py"
        spec = importlib.util.spec_from_file_location("pcb_calculator_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if original is None:
            sys.modules.pop("emsutil", None)
        else:
            sys.modules["emsutil"] = original


@pytest.fixture
def pcb(calc):
    dielectric = types.SimpleNamespace(er=4.2)
    return calc.PCBCalculator([0.0, 0.2], [dielectric], 0.001)


def test_microstrip_inverse_respects_user_bounds_and_round_trips(pcb):
    target = pcb.microstrip.z0(0.4)
    width = pcb.microstrip.w_for_z0(target, w_min=0.3, w_max=0.5)
    assert 0.3 <= width <= 0.5
    assert pcb.microstrip.z0(width) == pytest.approx(target, rel=1e-6)
    with pytest.raises(ValueError, match="outside the achievable range"):
        pcb.microstrip.w_for_z0(target, w_min=0.03, w_max=0.04)


def test_unattainable_target_is_not_silently_clamped(pcb):
    with pytest.raises(ValueError, match="outside the achievable range"):
        pcb.microstrip.w_for_z0(500.0)


def test_cpw_inverse_converts_mm_bounds_to_metres(pcb):
    target = pcb.cpw.z0(0.3, 0.15)
    width = pcb.cpw.w_for_z0(target, 0.15, w_min=0.1, w_max=1.0)
    assert width == pytest.approx(0.3, rel=1e-6)
    assert pcb.cpw.z0(width, 0.15) == pytest.approx(target, rel=1e-6)


def test_dispersion_matches_independent_qucs_coefficient(calc):
    # Qucs Kirschning/Jansen R8 contains both er**1.674 and fn**2.745
    # inside the exponential. This pinned value catches the 69ddfce refactor.
    z = calc.microstrip_z0_dispersion(0.2e-3, 0.2e-3, 4.2, 10e9)
    assert math.isfinite(z)
    assert z == pytest.approx(72.54344530474124, rel=1e-10)
    with pytest.raises(ValueError, match="published geometry/frequency range"):
        calc.microstrip_z0_dispersion(0.2e-3, 0.2e-3, 4.2, 200e9)


def test_missing_or_mixed_dielectric_is_not_treated_as_homogeneous(calc):
    missing = calc.PCBCalculator([0, 0.2], [], 0.001)
    with pytest.raises(ValueError, match="Missing dielectric"):
        missing.microstrip.z0(0.3)
    mixed = calc.PCBCalculator(
        [0, 0.1, 0.2],
        [types.SimpleNamespace(er=2.0), types.SimpleNamespace(er=10.0)],
        0.001,
    )
    with pytest.raises(ValueError, match="Mixed dielectric"):
        mixed.microstrip.z0(0.3)


def test_air_dispersion_keeps_positive_static_modal_limits(calc):
    single = calc.microstrip_z0(0.2e-3, 0.2e-3, 1.0)
    assert calc.microstrip_z0_dispersion(0.2e-3, 0.2e-3, 1.0, 1e9) == pytest.approx(single)
    static = calc.coupled_microstrip_z0_even_odd(0.2e-3, 0.2e-3, 0.2e-3, 1.0)
    dynamic = calc.coupled_microstrip_z0_even_odd(0.2e-3, 0.2e-3, 0.2e-3, 1.0, f=1e9)
    assert dynamic == pytest.approx(static)
    assert all(value > 0 for value in dynamic)


def test_broadside_matches_independent_cohn_reference(calc):
    zdiff, zcm = calc.broadside_stripline_zdiff_zcm(0.594, 0.234, 1.0, 2.2)
    assert float(zdiff) == pytest.approx(59.3999382784, rel=1e-8)
    assert float(zcm) == pytest.approx(44.0062304122, rel=1e-8)


def test_wide_stripline_has_no_elliptic_clipping_floor(calc):
    assert calc.stripline_z0(20.0, 1.0, 4.0) == pytest.approx(2.3037358463, rel=1e-8)
    assert calc.stripline_z0(100.0, 1.0, 4.0) == pytest.approx(0.4688440185, rel=1e-8)


def test_thick_microstrip_eeff_includes_air_width_factor(calc):
    assert calc.microstrip_eeff(0.2e-3, 0.2e-3, 4.2, 35e-6) == pytest.approx(
        2.8942522649, rel=1e-8
    )


def test_unqualified_differential_cpw_fails_closed(calc):
    with pytest.raises(NotImplementedError, match="coupled conformal or field solver"):
        calc.differential_cpw_zdiff_zcm(1e-3, 1e-3, 0.1e-3, 1e-3, 4.2)


def test_supported_microstrip_grid_has_finite_positive_impedances(calc):
    h = 0.2e-3
    for width_ratio, er, ghz in itertools.product(
        (0.1, 0.2, 0.5, 1, 2, 5, 10),
        (1, 1.01, 2.2, 4.2, 10, 18),
        (0.1, 1, 10, 50, 100),
    ):
        z = float(calc.microstrip_z0_dispersion(width_ratio * h, h, er, ghz * 1e9))
        assert math.isfinite(z) and 0 < z < 1000


def test_coupled_microstrip_grid_has_finite_positive_modes(calc):
    h = 0.2e-3
    for width_ratio, gap_ratio, er, ghz in itertools.product(
        (0.1, 0.5, 1, 2, 10),
        (0.1, 0.5, 1, 2, 10),
        (1, 1.01, 2.2, 4.2, 10, 18),
        (0.1, 1, 10, 50),
    ):
        even, odd = calc.coupled_microstrip_z0_even_odd(
            width_ratio * h, gap_ratio * h, h, er, f=ghz * 1e9
        )
        assert all(math.isfinite(z) and 0 < z < 1000 for z in (even, odd))


def test_stripline_rejects_nonphysical_geometry_instead_of_clipping(calc):
    with pytest.raises(ValueError, match="b > t"):
        calc.stripline_z0(0.2e-3, 0.2e-3, 4.2, t=0.2e-3)
    with pytest.raises(ValueError, match="positive"):
        calc.stripline_z0(-0.2e-3, 0.2e-3, 4.2)


def test_coax_cutoff_does_not_disguise_exact_solver_failures(calc, monkeypatch):
    inner, outer = 0.2e-3, 1e-3
    for mode, n, method in (("te", 1, calc.coax_cutoff_te),
                            ("tm", 0, calc.coax_cutoff_tm)):
        frequency = method(inner, outer, n=n)
        wave_number = 2.0 * math.pi * frequency / calc.C0
        residual = calc._coax_mode_char(mode, n, wave_number * inner / 2, outer / inner)
        assert abs(residual) < 1e-10
    with pytest.raises(ValueError, match="d_outer > d_inner"):
        calc.coax_cutoff_te(-0.2e-3, 1e-3)
    with pytest.raises(ValueError, match="supports only TE"):
        calc.coax_cutoff_te(0.2e-3, 1e-3, n=2, exact=False)
    with pytest.raises(ValueError, match="supports only TM"):
        calc.coax_cutoff_tm(0.2e-3, 1e-3, n=1, exact=False)
    def failed_root(**_kwargs):
        raise ValueError("modal root unavailable")
    monkeypatch.setattr(calc, "_coax_mode_root", failed_root)
    with pytest.raises(ValueError, match="modal root unavailable"):
        calc.coax_cutoff_te(0.2e-3, 1e-3)
    with pytest.raises(ValueError, match="modal root unavailable"):
        calc.coax_cutoff_tm(0.2e-3, 1e-3)


def test_twisted_pair_inverse_solves_twist_dependent_relation(calc):
    for twists in (0.0, 500.0, 10000.0):
        spacing = 0.55e-3
        wire = 0.2e-3
        z0 = calc.twisted_pair_z0(spacing, wire, 9.0, twists_per_len=twists)
        solved = calc.twisted_pair_d_center_for_z0(
            z0, wire, 9.0, twists_per_len=twists
        )
        assert solved == pytest.approx(spacing, rel=1e-10)
        assert calc.twisted_pair_z0(
            solved, wire, 9.0, twists_per_len=twists
        ) == pytest.approx(z0, rel=1e-10)
    with pytest.raises(ValueError, match="positive"):
        calc.twisted_pair_z0(0.5e-3, -0.2e-3, 4.0)
