"""Regression checks for stackup-unit inverse solvers and line dispersion.

The calculator uses Material only as a type annotation in these tests, so the
lightweight stub avoids importing EMerge's optional meshing/runtime stack.
"""

import importlib.util
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
