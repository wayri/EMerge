# PCB Calculator Formula Reference (Detailed)

## 1) Scope
This document is the detailed math reference for the implemented transmission-line calculators and inverse solvers.

- EMerge calculator implementation:
  - `EMerge/src/emerge/_emerge/geo/pcb_tools/calculator.py`


## 1.1) How To Validate Formulas In Practice
Use layered validation, from fastest to highest confidence:

1. Unit/regression checks (already in repo):
- Run:
```powershell
python -m unittest discover -s tests -v
```
- Confirms forward/inverse consistency, trend sanity, and known reference points.

2. Cross-tool numerical correlation:
- Compare identical geometry/material inputs against:
  - Qucs/QucsStudio transmission line calculators (where available) [R13][R14].
  - Independent calculator implementations (for overlapping models).
- Acceptance target for quasi-static closed forms: typically within low single-digit percent for supported ranges.

3. Literature spot checks:
- Reproduce published/example values from primary references (equation-level checks) [R1]-[R8], [R10]-[R12].

4. EM solver benchmarking (highest confidence):
- Sweep `W`, `S`, `G`, dielectric height, and `er` in 2.5D/3D EM.
- Extract `Z0`, `Zdiff`, `Zcm`, and compare versus model outputs.
- Use these comparisons to define validity envelopes per topology.

5. Range-gating:
- Enforce geometry constraints where formulas are known to become non-unique or inaccurate (for example broadside `g_for_zdiff` branch constraint).

## 2) Implementation Status
| Topology | Forward Model | Reverse Model | EMerge | pcb_libs |
|---|---|---|---|---|
| Microstrip | Implemented (static + dispersion) | `w_for_z0` | Yes | Yes |
| Stripline (centered) | Implemented (includes finite-`t` model) | `w_for_z0` | Yes | Yes |
| Edge-coupled stripline | Implemented | `w_for_zdiff`, `s_for_zdiff` | Yes | Yes |
| Broadside-coupled stripline | Implemented (full Cohn implicit-k model) | `w_for_zdiff`, `g_for_zdiff` | Yes | Yes |
| CPW | Implemented (static + dispersion) | `w_for_z0` | Yes | Yes |
| GCPW | Implemented (static + dispersion) | `w_for_z0` | Yes | Yes |
| Edge-coupled microstrip | Implemented (static + dispersion) | `w_for_zdiff`, `s_for_zdiff` | Yes | Yes |
| Differential CPW / DCPWG | Implemented (fed by dispersive CPW core) | `w_for_zdiff`, `s_for_zdiff` | Yes | Yes |
| Coax | Implemented | `d_inner_for_z0` | Yes | Yes |
| Twisted pair | Implemented | `d_center_for_z0`, `d_wire_for_z0` | Yes | Yes |
| Rectangular waveguide | Implemented | `a_for_fc`, `a_for_z_te10` | Yes | Yes |

## 3) Symbols, Constants, And Units
### 3.1 Constants
- `c0 = 299792458.0` m/s
- `mu0 = 4*pi*1e-7` H/m
- `Zf0 = n0 = 376.730313668...` Ohm (free-space impedance)
- `pi = 3.14159...`
- `tau = 2*pi`

### 3.2 Geometry Symbols
- `W`: conductor width
- `S`: lateral slot or edge spacing
- `G`: broadside vertical spacing between coupled strips
- `h` or `th`: dielectric height to reference plane
- `b`: stripline cavity height (ground-to-ground spacing)
- `t`: conductor thickness
- `a`, `b` (waveguide): broad and narrow wall dimensions

### 3.3 Material Symbols
- `er`: relative permittivity
- `mur`: relative permeability
- `tand`: dielectric loss tangent

### 3.4 Mode Symbols
- `Z0`: characteristic impedance (single-ended)
- `Z0e`, `Z0o`: even and odd mode impedances
- `Zdiff = 2*Z0o`
- `Zcm = Z0e/2`

## 4) Shared Mathematical Primitives
### 4.1 Complete Elliptic Integral Using AGM
Implemented helper:

`K(k) = pi / (2 * AGM(1, sqrt(1-k^2)))`, for `0 <= k < 1`.

Reference: elliptic-integral definitions and properties [R11].

### 4.2 Elliptic Ratio
Implemented helper:

`R(k) = K(k) / K(kp)`, where `kp = sqrt(1-k^2)`.

### 4.3 Hyperbolic Helpers
- `sech(x) = 1/cosh(x)`
- `coth(x) = cosh(x)/sinh(x)` with numerical guard near zero.

## 5) Shared Inverse Solver
Inverse APIs use a common strategy:

1. Log-spaced sweep:
- `x_i` sampled by `geomspace(x_min, x_max, n)`.

2. Interpolation estimate:
- Compute `y_i = f(x_i)`.
- Infer monotonic direction.
- Interpolate `x_est` for target `y_target`.

3. Bracket + bisection refinement:
- Detect adjacent sample pair bracketing target.
- Refine by bisection (up to 64 iters).
- Return fallback interpolation value if bracketing fails.

This is implemented as `_scan_inverse(...)` and is used by `w_for_*`/`s_for_*`/`g_for_*`.

## 6) Layer Effective Permittivity In Stackup APIs
For stackup-based APIs, `effective_er(...)` is computed as thickness-weighted average between selected layers:

`er_eff = sum(er_i * d_i) / sum(d_i)`.

If explicit `er` is provided in call args, it overrides stackup averaging.

## 7) Microstrip (Single-Ended)
Primary basis: Hammerstad/Jensen family with standard CAD adaptations [R1], [R7], [R13].
### 7.1 Forward
Define normalized width:

`u = W/h`.

Finite thickness correction (if `t > 0`):

- `thn = t/h`
- `x = sqrt(6.517*u)`
- `du1 = (thn/pi) * ln(1 + (4*e)/(thn*coth(x)^2))`
- `dur = 0.5*du1*(1 + sech(sqrt(max(er-1,0))))`
- `u <- u + dur`

Effective permittivity:

`eeff = (er+1)/2 + (er-1)/(2*sqrt(1+12/u))`

Narrow-line correction (`u < 1`):

`eeff += 0.04*(1-u)^2*(er-1)/2`.

Impedance:

- If `u <= 1`:
  - `Z0 = (60/sqrt(eeff))*ln(8/u + 0.25*u)`
- Else:
  - `Z0 = (120*pi/sqrt(eeff)) / (u + 1.393 + 0.667*ln(u+1.444))`

### 7.2 Reverse
- `w_for_z0(...)` solves `microstrip_z0(W)=Z0_target` with `_scan_inverse`.

### 7.3 Frequency-Dependent Effective Permittivity (Kirschning/Jansen)
With `f_n = f*h/1e6` (`f` in Hz, `h` in m):

- `P1 = 0.27488 + u*(0.6315 + 0.525/(1+0.0157*f_n)^20) - 0.065683*exp(-8.7513*u)`
- `P2 = 0.33622*(1-exp(-0.03442*er))`
- `P3 = 0.0363*exp(-4.6*u)*(1-exp(-(f_n/38.7)^4.97))`
- `P4 = 1 + 2.751*(1-exp(-(er/15.916)^8))`
- `P = P1*P2*((P3*P4 + 0.1844)*f_n)^1.5763`
- `eeff(f) = er - (er - eeff(0))/(1 + P)`

### 7.4 Frequency-Dependent Impedance (Kirschning/Jansen)
Using the Qucs transcalc form:

- `Z0(f) = Z0(0) * D`
- `D = (R13/R14)^R17`

Where `R1..R17` follow the coefficient set from [R15].

## 8) Stripline (Centered)
Primary basis: Cohn shielded stripline [R2], summary treatment [R12].
### 8.1 Forward
- `x = pi*W/(2*b)`
- `k = sech(x)`
- `Z0 = (Zf0/(4*sqrt(er))) * R(k)`

Finite-thickness branch (`t>0`) uses the stripline equation set from Qucs transcalc [R17]:

- `x = t/b`
- `m = 2/(1 + (2x/3)*(1-x))`
- `Bc = (x/(pi*(1-x))) * {1 - 0.5*ln[(x/(2-x))^2 + (0.0796*x/(W/b + 1.1x))^m]}`
- `A = 1/(W/(b-t) + Bc)`
- `Z0 = (30/sqrt(er))*ln(1 + (4/pi)*A*((8/pi)*A + sqrt(((8/pi)*A)^2 + 6.27)))`

### 8.2 Reverse
- `w_for_z0(...)` solves `stripline_z0(W)=Z0_target`.

## 9) Edge-Coupled Stripline
Primary basis: Cohn coupled shielded stripline [R3], summary treatment [R12].
### 9.1 Forward
Odd-mode modulus transform:

- `x1 = pi*W/(2*b)`
- `x2 = pi*(W+S)/(2*b)`
- `k0p = tanh(x1)*coth(x2)`
- `k0 = sqrt(1-k0p^2)`

Odd-mode impedance:

`Z0o = (Zf0/(4*sqrt(er))) * R(k0)`.

Differential:

`Zdiff = 2*Z0o`.

### 9.2 Reverse
- `w_for_zdiff(...)` and `s_for_zdiff(...)` via `_scan_inverse`.

## 10) Broadside-Coupled Stripline (Full Cohn Model)
Primary basis: Cohn broadside-coupled stripline [R4], equation summary and design context [R12].
### 10.1 Forward
Definitions:

- `s = G/b`
- `wbar = W/b`
- `0 < s < 1`
- Solve for `k` in `(s,1)`.

Implicit width relation:

- `Rk = sqrt((k-s)/(1/k - s))`
- `wbar = (2/pi)*atanh(Rk) - s*atanh(Rk/k)`

Even-mode:

`Z0e = (188.3/sqrt(er)) * K(kp)/K(k)`, `kp = sqrt(1-k^2)`.

Odd-mode:

`Z0o = (296.1*s) / (sqrt(er)*atanh(k))`.

Mode conversion:

- `Zdiff = 2*Z0o`
- `Zcm = Z0e/2`

### 10.2 Inverse
- `w_for_zdiff(...)`: solve width for target `Zdiff`.
- `g_for_zdiff(...)`: solve gap for target `Zdiff`.

Important branch note:
- `Zdiff(G)` is non-monotonic globally.
- Current inverse implementation constrains `g_for_zdiff` to monotonic branch:
  - `G < 0.5*b`.

## 11) CPW And GCPW
Primary basis: conformal mapping models from Ghione/Naldi and related CPW literature [R5], [R6], as implemented in Qucs transcalc style [R13].
Let:

- `a = W`
- `b = W + 2*S`
- `k1 = a/b`
- `q1 = R(k1)`.

### 11.1 CPW (no backside metal)
- `k2 = sinh((pi/4)*a/h) / sinh((pi/4)*b/h)`
- `q2 = R(k2)`
- `eeff = 1 + ((er-1)/2)*(q2/q1)`
- `zr = Zf0/(4*q1)`
- `Z0 = zr/sqrt(eeff)`

### 11.2 GCPW (with backside metal)
- `k3 = tanh((pi/4)*a/h) / tanh((pi/4)*b/h)`
- `q3 = R(k3)`
- `qz = 1/(q1+q3)`
- `eeff = 1 + q3*qz*(er-1)`
- `zr = (Zf0/2)*qz`
- `Z0 = zr/sqrt(eeff)`

### 11.3 Finite Thickness Correction (both branches)
If `t > 0`:

- `d = (1.25*t/pi) * (1 + ln(4*pi*W/t))`
- `ke = k1 + (1-k1^2)*d/(2*S)`
- `qe = R(ke)`

Impedance factor update:
- CPW: `zr = Zf0/(4*qe)`
- GCPW: `qz = 1/(qe+q3)`, `zr = (Zf0/2)*qz`

Effective permittivity update:

`eeff <- eeff - [0.7*(eeff-1)*t/S] / [q1 + 0.7*t/S]`.

### 11.4 Frequency Dispersion (Qucs transcalc form)
Define:

- `f_te = (c0/4)/(h*sqrt(er-1))`
- `p = ln(W/h)`
- `u = 0.54 - (0.64 - 0.015*p)*p`
- `v = 0.43 - (0.86 - 0.54*p)*p`
- `G = exp(u*ln(W/S) + v)`
- `sqrt(eeff(f)) = sqrt(eeff(0)) + (sqrt(er)-sqrt(eeff(0))) / (1 + G*(f/f_te)^(-1.8))`

And impedance is updated via:

- `Z0(f) = Z0(0) * sqrt(eeff(0)/eeff(f))`

### 11.5 Reverse
- `w_for_z0(...)` for both CPW and GCPW.

## 12) Edge-Coupled Differential Microstrip (Kirschning/Jansen)
Primary basis: Kirschning/Jansen coupled microstrip modeling lineage [R8], with implementation form consistent with Qucs formulations [R14].
### 12.1 Base Variables
- `u = W/h`
- `g = S/h`

### 12.2 Thickness Adjustment (code-equivalent branch logic)
If conditions are met, compute corrected `ue`, `uo` using:

- `dw = t*(1 + ln(2*h/t))/pi` or `dw = t*(1 + ln(4*pi*W/t))/pi` branch
- `dt = 2*t*h/(S*er)`
- `we = W + dw*(1 - 0.5*exp(-0.69*dw/dt))`
- `wo = we + dt`
- `ue = we/h`, `uo = wo/h`.

### 12.3 Even/Odd Permittivity And Impedance Terms
- `v = ue*(20+g^2)/(10+g^2) + g*exp(-g)`
- `er_eff_e = microstrip_eeff(v*h, h, er)`
- `er_eff = microstrip_eeff(uo*h, h, er)`
- `d = 0.593 + 0.694*exp(-0.562*uo)`
- `b_o = 0.747*er/(0.15+er)`
- `c_o = b_o - (b_o-0.207)*exp(-0.414*uo)`
- `a_o = 0.7287*(er_eff - (er+1)/2)*(1-exp(-0.179*uo))`
- `er_eff_o = ((er+1)/2 + a_o - er_eff)*exp(-c_o*g^d) + er_eff`

Coupling coefficients:
- `q1 = 0.8695*u^0.194`
- `q2 = 1 + 0.7519*g + 0.189*g^2.31`
- `q3 = 0.1975 + (16.6 + (8.4/g)^6)^(-0.387) + (1/241)*ln(g^10/(1+(g/3.4)^10))`
- `q4 = (2*q1/q2) / (exp(-g)*u^q3 + (2-exp(-g))*u^(-q3))`
- `q5 = 1.794 + 1.14*ln(1 + 0.638/(g + 0.517*g^2.43))`
- `q6 = 0.2305 + (1/281.3)*ln(g^10/(1+(g/5.8)^10)) + (1/5.1)*ln(1+0.598*g^1.154)`
- `q7 = (10+190*g^2)/(1+82.3*g^3)`
- `q8 = exp(-6.5 - 0.95*ln(g) - (g/0.15)^5)`
- `q9 = ln(q7)*(q8 + 1/16.5)`
- `q10 = (q2*q4 - q5*exp(ln(u)*q6*u^(-q9))) / q2`

Let `zl1 = microstrip_z0(W,h,er,t=0)`:

- `Zeven = sqrt(er_eff/er_eff_e) * zl1 / (1 - zl1*sqrt(er_eff)*q4/Zf0)`
- `Zodd  = sqrt(er_eff/er_eff_o) * zl1 / (1 - zl1*sqrt(er_eff)*q10/Zf0)`

Mode conversion:
- `Zdiff = 2*Zodd`
- `Zcm = Zeven/2`

### 12.4 Reverse
- `w_for_zdiff(...)`
- `s_for_zdiff(...)`

Both are solved numerically with sampled evaluations.

### 12.5 Frequency-Dependent Extension
The implementation includes the Kirschning/Jansen dispersion blocks for coupled microstrip (even and odd mode):

- `eeff,e(f)` and `eeff,o(f)` via `P1..P15`, `Fe`, `Fo` coefficient sets.
- `Z0e(f)` via `Ce`, `de`, `Q0`, `R1..R16`.
- `Z0o(f)` via `Q22..Q29`.

Coefficient forms follow the Qucs transcalc implementation [R16].

## 13) Differential CPW / DCPWG
Primary basis:
- CPW/GCPW capacitance/impedance from conformal mapping [R5], [R6], [R13].
- Transmission-line relation between per-unit-length parameters and characteristic impedance [R10].
### 13.1 Construction
Built from the existing CPW/GCPW core by per-unit-length capacitance decomposition.
If frequency is provided, the CPW/GCPW dispersive forms from Section 11.4 are used in the capacitance extraction.

From single-line CPW result:
- `C = sqrt(eeff)/(c0*Z0)`
- `C_air = 1/(c0*Z0_air)` where `Z0_air` is CPW solved with `er=1`.

Compute:
- Outer-slot contribution with `(W, S_ground)`: `C_out`, `C_air_out`, `Z_out`
- Inner pair-slot contribution approximated by `(W, S_pair/2)`: `C_in`, `C_air_in`

Odd mode:
- `C_odd = C_out + C_in`
- `C_air_odd = C_air_out + C_air_in`
- `Zodd = 1/(c0*sqrt(C_odd*C_air_odd))`

Even mode:
- `Zeven = Z_out`

Final:
- `Zdiff = 2*Zodd`
- `Zcm = Zeven/2`

Applies for:
- differential CPW (no backside metal)
- DCPWG (with backside metal)

### 13.2 Reverse
- `w_for_zdiff(...)`
- `s_for_zdiff(...)`

## 14) Coax
Primary basis: standard coax TEM line formulas [R10].
### 14.1 Forward
- `Z0 = (Zf0/(2*pi*sqrt(er))) * ln(d_outer/d_inner)`

### 14.2 Reverse
- `d_inner = d_outer / exp(2*pi*sqrt(er)*Z0/Zf0)`

### 14.3 Cutoff (Exact Modal Solver)
Exact modal cutoffs are implemented via annular-waveguide eigenvalue roots:

Let `a = d_inner/2`, `b = d_outer/2`, `k_c = 2*pi*f_c*sqrt(er*mur)/c0`.

TM modes (`E_z` non-zero) satisfy:

- `J_n(k_c a) Y_n(k_c b) - Y_n(k_c a) J_n(k_c b) = 0`

TE modes (`H_z` non-zero) satisfy:

- `J'_n(k_c a) Y'_n(k_c b) - Y'_n(k_c a) J'_n(k_c b) = 0`

Where:
- `J'_n(x) = 0.5*(J_{n-1}(x) - J_{n+1}(x))`
- `Y'_n(x) = 0.5*(Y_{n-1}(x) - Y_{n+1}(x))`

The implementation scans for bracketing sign changes and refines each root by bisection.
Default `cutoffs()` returns exact `TE11` and `TM01` cutoffs; approximation fallback is retained if runtime Bessel backend is unavailable.

## 15) Twisted Pair
Primary basis:
- `Z0` two-wire/quasi-static `acosh(D/d)` form [R10].
- Implemented `eeff` twist correction follows Qucs technical documentation form [R9], [R14].
### 15.1 Effective Permittivity
Require `d_center > d_wire`.

- `theta = atan(twists_per_len*pi*d_center)`
- `q = 0.001` if `ptfe=True`, else `q = 0.25 + 0.0004*theta^2`
- `eeff = er1 + q*(er-er1)`

### 15.2 Impedance
- `Z0 = Zf0/(pi*sqrt(eeff)) * acosh(d_center/d_wire)`

### 15.3 Reverse
- `d_center_for_z0`: `d_center = d_wire * cosh(pi*Z0*sqrt(eeff)/Zf0)`
- `d_wire_for_z0`: `d_wire = d_center / cosh(pi*Z0*sqrt(eeff)/Zf0)`

## 16) Rectangular Waveguide
Primary basis: standard waveguide field/cutoff/impedance relations [R10].
### 16.1 Cutoff
For mode `(m,n)`:

`fc = (c0/(2*sqrt(er*mur))) * sqrt((m/a)^2 + (n/b)^2)`.

### 16.2 Propagation
For frequency `f`:

- `k = 2*pi*f*sqrt(er*mur)/c0`
- `kc = 2*pi*fc*sqrt(er*mur)/c0`
- If `f <= fc`, `beta = 0`
- Else `beta = sqrt(k^2-kc^2)`

### 16.3 Wave Impedance
- TE: `Zte = Zf0*sqrt(mur/er)/sqrt(1-(fc/f)^2)` for `f > fc`, else `inf`
- TM: `Ztm = Zf0*sqrt(mur/er)*sqrt(1-(fc/f)^2)` for `f > fc`, else `0`

### 16.4 Guided Wavelength
- `lambda_g = 2*pi/beta` for `beta>0`, else `inf`.

### 16.5 Closed-Form Inverses
- From cutoff:
  - `a = m*c0/(2*fc*sqrt(er*mur))`
- TE10 impedance design:
  - `q = Zf0*sqrt(mur/er)/Z_target`
  - `fc = f*sqrt(1-q^2)`
  - then `a = c0/(2*fc*sqrt(er*mur))`

## 17) API-Level Reverse Methods
Implemented reverse methods across namespaces:

- `microstrip.w_for_z0`
- `stripline.w_for_z0`
- `edge_coupled_stripline.w_for_zdiff`
- `edge_coupled_stripline.s_for_zdiff` (EMerge)
- `broadside_coupled_stripline.w_for_zdiff`
- `broadside_coupled_stripline.g_for_zdiff`
- `cpw.w_for_z0`
- `gcpw.w_for_z0`
- `edge_coupled_microstrip.w_for_zdiff`
- `edge_coupled_microstrip.s_for_zdiff`
- `coplanar_microstrip_diff.w_for_zdiff`
- `coplanar_microstrip_diff.s_for_zdiff`
- `dcpwg.w_for_zdiff`
- `dcpwg.s_for_zdiff`
- `coax.d_inner_for_z0`
- `twisted_pair.d_center_for_z0`
- `twisted_pair.d_wire_for_z0`
- `rectangular_waveguide.a_for_fc`
- `rectangular_waveguide.a_for_z_te10`

## 18) Validation Coverage
### 18.1 Test Command
```powershell
python -m unittest discover -s tests -v
```

### 18.1.1 CSV Validation Command
```powershell
python tests/validate_formulas.py
```

Generated artifacts:
- `tests/validation_outputs/all_checks.csv`
- `tests/validation_outputs/failing_checks.csv`
- `tests/validation_outputs/summary_by_topology.csv`
- `tests/validation_outputs/summary_overview.csv`

### 18.2 Current Result
- `25 tests, 25 passed`.
- CSV validation: `70 checks, 70 passed`.

### 18.3 Covered Behaviors
- Microstrip forward/reverse consistency.
- Microstrip dispersion trend checks (`eeff(f)`, `Z0(f)`).
- Stripline forward/reverse consistency.
- Stripline finite-thickness trend check.
- Broadside stripline forward/reverse consistency + trend.
- CPW forward/reverse consistency.
- CPW dispersion trend checks.
- Differential CPW/DCPWG forward/reverse consistency + trend.
- Edge-coupled microstrip weak-coupling sanity.
- Edge-coupled microstrip forward/reverse consistency.
- Coax formula + closed-form inverse + cutoff sanity.
- Coax exact TE11/TM01 eigen-root residual checks.
- Twisted pair eeff/impedance sanity + inverse geometry checks.
- Rectangular waveguide TE10 reference checks.
- Coarse-grid inverse refinement regression checks.

## 19) Accuracy And Limit Notes
- Most models are quasi-static; explicit dispersion support is implemented for microstrip, coupled microstrip, CPW, and GCPW.
- CPW and coupled-line expressions are closed-form approximations, not full-wave solvers.
- Broadside `g_for_zdiff` inverse is branch-limited (`G < 0.5*b`) because the global mapping is non-monotonic.
- Coax modal cutoffs are solved from annular-waveguide eigen equations when runtime Bessel backend is available; otherwise approximation fallback is used.

## 20) Worked Usage Guide (Layers, Materials, Options, And Examples)
All geometry is shown in `mm` and all impedance in `Ohm` unless explicitly noted.
All API examples target `EMerge/src/emerge/_emerge/geo/pcb_tools/calculator.py`.

### 20.1) How Stackup Definition Works
`PCBCalculator(layers, materials, unit)`:

- `layers`: z-coordinates in your chosen geometry unit.
- `materials`: dielectric entries for each interval between layers.
- `unit`: conversion from geometry unit to meters.

Rules:

- `len(materials) == len(layers) - 1`
- `materials[i]` applies to interval `(layers[i], layers[i+1])`
- layer indices can be positive (`0,1,2,...`) or negative (`-1` is last layer)

Typical `unit` values:

- `1e-3` if geometry is provided in `mm`
- `1e-6` if geometry is provided in `um`
- `1.0` if geometry is already in `m`

### 20.2) Material Objects (Constant Or Frequency-Dependent)
The calculator reads `material.er` via:

- float: constant dielectric constant
- callable: `er(f0)` returning dielectric constant at frequency `f0`
- object with method: `er.scalar(f0)`

```python
import numpy as np
from emerge._emerge.geo.pcb_tools.calculator import PCBCalculator

class MatConst:
    def __init__(self, er):
        self.er = float(er)

class MatDisp:
    # Example only: simplistic frequency dependence
    def __init__(self, er_1ghz=3.8):
        self.er_1ghz = float(er_1ghz)
    def er(self, f0):
        return self.er_1ghz + 0.03 * np.log10(max(f0, 1.0) / 1e9)
```

### 20.3) Layer Cookbook (Common PCB Arrangements)
Microstrip-style:

Scenario: Two-layer PCB model with one dielectric slab where top trace references bottom ground.
```python
# Signal on top layer index -1, ground on layer 0
calc_ms = PCBCalculator(np.array([0.0, 1.6]), [MatConst(4.2)], 1e-3)
```

Stripline cavity:

Scenario: Symmetric stripline cavity with ground planes above and below the signal layer region.
```python
# Ground planes at layer indices 0 and 2, strip trace centered between them
calc_sl = PCBCalculator(np.array([0.0, 0.2, 0.4]), [MatConst(3.5), MatConst(3.5)], 1e-3)
```

CPW/GCPW substrate:

Scenario: Single substrate thickness used for CPW and GCPW calculations referenced to the backside plane.
```python
calc_cp = PCBCalculator(np.array([0.0, 0.8]), [MatConst(3.2)], 1e-3)
```

Nonuniform dielectric between signal and reference:

Scenario: Multilayer stackup with mixed dielectrics to demonstrate thickness-weighted effective `er`.
```python
# Effective er is thickness-weighted across intervals between selected layers
calc_mix = PCBCalculator(
    np.array([0.0, 0.15, 0.50, 0.80]),
    [MatConst(3.3), MatConst(4.1), MatConst(2.9)],
    1e-3,
)
```

### 20.4) Global Option Meanings
Options reused by many APIs:

- `f0`: frequency in `Hz`; used for frequency-dependent dielectric and dispersive line models.
- `er`: override dielectric constant; if provided, stackup-weighted material extraction is bypassed.
- `t`: conductor thickness in your geometry unit.
- inverse options: `w_min`, `w_max`, `s_min`, `s_max`, `g_min`, `g_max`, `n`
- `n`: sample count for inverse scan (`_scan_inverse`), default `501`.
- higher `n` improves robustness/accuracy but costs runtime.

### 20.5) Microstrip API (All Options)
Signatures:

```python
microstrip.z0(w, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0)
microstrip.eeff(w, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0)
microstrip.w_for_z0(Z0, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=501)
microstrip.quarter_wave(f, layer=-1, ground_layer=0, f0=None, w=None, Z0=50.0, t=0.0)
```

Examples:

Scenario: Compute microstrip impedance/effective permittivity, solve inverse width for 50 Ohm, then compute quarter-wave length.
```python
# Forward Z0 and eeff
z0 = calc_ms.microstrip.z0(0.35, layer=-1, ground_layer=0, f0=1e9, t=0.035)   # 123.550603
ee = calc_ms.microstrip.eeff(0.35, layer=-1, ground_layer=0, f0=1e9, t=0.035)

# Inverse width with explicit search range
w = calc_ms.microstrip.w_for_z0(50.0, layer=-1, ground_layer=0, f0=2e9, t=0.018, w_min=0.05, w_max=3.0, n=1201)

# Override dielectric directly (ignores stackup-material extraction)
w_er = calc_ms.microstrip.w_for_z0(50.0, er=3.66, t=0.018)

# Quarter-wave length from Z0 target
l_qw = calc_ms.microstrip.quarter_wave(f=2.4e9, Z0=50.0, t=0.035)
```

### 20.6) Stripline API (All Options)
Signatures:

```python
stripline.z0(w, gnd_top, gnd_bot, f0=1e9, er=None, t=0.0)
stripline.w_for_z0(Z0, gnd_top, gnd_bot, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=501)
```

Examples:

Scenario: Centered stripline design between two ground layers, including finite copper thickness.
```python
z0 = calc_sl.stripline.z0(0.24, gnd_top=2, gnd_bot=0, f0=1e9, t=0.035)        # 41.1496412
w  = calc_sl.stripline.w_for_z0(50.0, gnd_top=2, gnd_bot=0, t=0.018, w_min=0.05, w_max=1.5, n=1001)
```

### 20.7) Edge-Coupled Stripline API (All Options)
Signatures:

```python
edge_coupled_stripline.zodd(w, s, gnd_top, gnd_bot, f0=1e9, er=None)
edge_coupled_stripline.zdiff(w, s, gnd_top, gnd_bot, f0=1e9, er=None)
edge_coupled_stripline.w_for_zdiff(Zdiff, s, gnd_top, gnd_bot, f0=1e9, er=None, w_min=1e-6, w_max=1e-1, n=501)
edge_coupled_stripline.s_for_zdiff(Zdiff, w, gnd_top, gnd_bot, f0=1e9, er=None, s_min=1e-6, s_max=1e-1, n=501)
```

Examples:

Scenario: Edge-coupled stripline pair where spacing is fixed for one solve and width is fixed for the other.
```python
zodd = calc_sl.edge_coupled_stripline.zodd(0.18, 0.12, gnd_top=2, gnd_bot=0)
zd   = calc_sl.edge_coupled_stripline.zdiff(0.18, 0.12, gnd_top=2, gnd_bot=0)                 # 96.8916683
w    = calc_sl.edge_coupled_stripline.w_for_zdiff(zd, s=0.12, gnd_top=2, gnd_bot=0, n=801)   # 0.18
s    = calc_sl.edge_coupled_stripline.s_for_zdiff(zd, w=0.18, gnd_top=2, gnd_bot=0, n=801)   # 0.12
```

### 20.8) Broadside-Coupled Stripline API (All Options)
Signatures:

```python
broadside_coupled_stripline.zdiff_zcm(w, g, gnd_top, gnd_bot, f0=1e9, er=None)
broadside_coupled_stripline.w_for_zdiff(Zdiff, g, gnd_top, gnd_bot, f0=1e9, er=None, w_min=1e-6, w_max=1e-1, n=501)
broadside_coupled_stripline.g_for_zdiff(Zdiff, w, gnd_top, gnd_bot, f0=1e9, er=None, g_min=1e-6, g_max=1e-1, n=501)
```

Notes:

- `g_for_zdiff` uses a monotonic-branch restricted solve (`g < ~0.5*b`) by design.

Examples:

Scenario: Broadside-coupled stripline pair with target differential impedance and controlled broadside gap branch.
```python
zd, zc = calc_sl.broadside_coupled_stripline.zdiff_zcm(0.18, 0.08, gnd_top=2, gnd_bot=0)    # 41.9240218, 35.8843103
w      = calc_sl.broadside_coupled_stripline.w_for_zdiff(zd, g=0.08, gnd_top=2, gnd_bot=0)
g      = calc_sl.broadside_coupled_stripline.g_for_zdiff(zd, w=0.18, gnd_top=2, gnd_bot=0, g_min=0.01, g_max=0.19, n=1001)
```

### 20.9) CPW API (All Options)
Signatures:

```python
cpw.z0(w, s, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0)
cpw.eeff(w, s, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0)
cpw.w_for_z0(Z0, s, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=501)
```

Examples:

Scenario: CPW line with given slot and thickness, including dispersive behavior at `f0`.
```python
z0 = calc_cp.cpw.z0(0.60, 0.20, layer=-1, ref_layer=0, f0=5e9, t=0.018)         # 69.5033424
ee = calc_cp.cpw.eeff(0.60, 0.20, layer=-1, ref_layer=0, f0=5e9, t=0.018)
w  = calc_cp.cpw.w_for_z0(50.0, s=0.20, layer=-1, ref_layer=0, f0=5e9, t=0.018, w_min=0.05, w_max=2.0, n=1201)
```

### 20.10) GCPW API (All Options)
`gcpw` has the same signatures as `cpw` and adds backside metal in the model.

Examples:

Scenario: Grounded CPW variant (backside metal included) for the same surface geometry as CPW.
```python
z0 = calc_cp.gcpw.z0(0.60, 0.20, f0=5e9, t=0.018)                                  # 66.4265730
w  = calc_cp.gcpw.w_for_z0(50.0, s=0.20, f0=5e9, t=0.018, w_min=0.05, w_max=2.0)
```

### 20.11) Edge-Coupled Microstrip API (All Options)
Signatures:

```python
edge_coupled_microstrip.even_odd(w, s, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0)
edge_coupled_microstrip.zdiff_zcm(w, s, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0)
edge_coupled_microstrip.w_for_zdiff(Zdiff, s, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=501)
edge_coupled_microstrip.s_for_zdiff(Zdiff, w, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0, s_min=1e-6, s_max=1e-1, n=501)
```

Examples:

Scenario: Differential edge-coupled microstrip where modal and differential/common-mode impedances are extracted.
```python
ze, zo = calc_ms.edge_coupled_microstrip.even_odd(0.30, 0.20, f0=10e9, t=0.035)
zd, zc = calc_ms.edge_coupled_microstrip.zdiff_zcm(0.30, 0.20, f0=10e9, t=0.035)            # 122.431623, 103.969822
w      = calc_ms.edge_coupled_microstrip.w_for_zdiff(zd, s=0.20, f0=10e9, t=0.035, n=801)  # 0.30
s      = calc_ms.edge_coupled_microstrip.s_for_zdiff(zd, w=0.30, f0=10e9, t=0.035, n=801)  # 0.20
```

### 20.12) Differential CPW APIs (Coplanar Microstrip Diff + DCPWG)
Signatures (both APIs):

```python
coplanar_microstrip_diff.zdiff_zcm(w, s_pair, s_ground, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0)
coplanar_microstrip_diff.w_for_zdiff(Zdiff, s_pair, s_ground, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=501)
coplanar_microstrip_diff.s_for_zdiff(Zdiff, w, s_ground, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0, s_min=1e-6, s_max=1e-1, n=501)

dcpwg.zdiff_zcm(w, s_pair, s_ground, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0)
dcpwg.w_for_zdiff(Zdiff, s_pair, s_ground, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=501)
dcpwg.s_for_zdiff(Zdiff, w, s_ground, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0, s_min=1e-6, s_max=1e-1, n=501)
```

Examples:

Scenario: Compare differential CPW without backside ground versus DCPWG with backside ground for the same pair geometry.
```python
# Coplanar microstrip differential (no backside metal)
zd, zc = calc_cp.coplanar_microstrip_diff.zdiff_zcm(0.35, 0.20, 0.25, f0=10e9, t=0.018)                 # 71.8965207, 43.2731876
w      = calc_cp.coplanar_microstrip_diff.w_for_zdiff(zd, s_pair=0.20, s_ground=0.25, f0=10e9, t=0.018)
s      = calc_cp.coplanar_microstrip_diff.s_for_zdiff(zd, w=0.35, s_ground=0.25, f0=10e9, t=0.018)

# DCPWG (with backside metal)
zd2, zc2 = calc_cp.dcpwg.zdiff_zcm(0.35, 0.20, 0.25, f0=10e9, t=0.018)                                   # 73.6760641, 42.4601301
w2       = calc_cp.dcpwg.w_for_zdiff(zd2, s_pair=0.20, s_ground=0.25, f0=10e9, t=0.018)
s2       = calc_cp.dcpwg.s_for_zdiff(zd2, w=0.35, s_ground=0.25, f0=10e9, t=0.018)
```

### 20.13) Coax API (All Options)
Signatures:

```python
coax.z0(d_inner, d_outer, er=1.0)
coax.d_inner_for_z0(Z0, d_outer, er=1.0)
coax.cutoff_te(d_inner, d_outer, er=1.0, mur=1.0, n=1, m=1, exact=True)
coax.cutoff_tm(d_inner, d_outer, er=1.0, mur=1.0, n=0, m=1, exact=True)
coax.cutoffs(d_inner, d_outer, er=1.0, mur=1.0, exact=True)
```

Options:

- `exact=True`: solves TE/TM cutoff roots from Bessel eigen-equations (preferred).
- `exact=False`: uses approximation fallback formulas.
- `n`, `m`: modal indices for TE/TM cutoff calls.

Examples:

Scenario: Coax line design using forward impedance, inverse inner-diameter solve, and modal cutoff queries.
```python
z0 = calc_cp.coax.z0(0.50, 2.00, er=2.1)                                                         # 57.3583313
di = calc_cp.coax.d_inner_for_z0(z0, d_outer=2.00, er=2.1)                                      # 0.50
fte = calc_cp.coax.cutoff_te(0.50, 2.00, er=2.1, mur=1.0, n=1, m=1, exact=True)
ftm = calc_cp.coax.cutoff_tm(0.50, 2.00, er=2.1, mur=1.0, n=0, m=1, exact=True)
fte11, ftm01 = calc_cp.coax.cutoffs(0.50, 2.00, er=2.1, mur=1.0, exact=True)                     # 5.41460132e10, 1.34917975e11 Hz
```

### 20.14) Twisted Pair API (All Options)
Signatures:

```python
twisted_pair.eeff(d_center, d_wire, er, er1=1.0, twists_per_len=0.0, ptfe=False)
twisted_pair.z0(d_center, d_wire, er, er1=1.0, twists_per_len=0.0, ptfe=False)
twisted_pair.d_center_for_z0(z0, d_wire, er, er1=1.0, twists_per_len=0.0, ptfe=False)
twisted_pair.d_wire_for_z0(z0, d_center, er, er1=1.0, twists_per_len=0.0, ptfe=False)
```

Options:

- `er`: bulk dielectric.
- `er1`: reference dielectric term used by the implemented Qucs-style correction.
- `twists_per_len`: turns per geometry unit.
- `ptfe=True`: uses PTFE-specific correction branch.

Examples:

Scenario: Twisted-pair impedance and geometry back-solves for a chosen dielectric and twist setting.
```python
ee = calc_cp.twisted_pair.eeff(d_center=2.0, d_wire=1.0, er=2.5, er1=1.0, twists_per_len=0.0, ptfe=False)
z0 = calc_cp.twisted_pair.z0(d_center=2.0, d_wire=1.0, er=2.5, er1=1.0, twists_per_len=0.0, ptfe=False)  # 134.67942
dc = calc_cp.twisted_pair.d_center_for_z0(z0, d_wire=1.0, er=2.5, er1=1.0, twists_per_len=0.0, ptfe=False)
dw = calc_cp.twisted_pair.d_wire_for_z0(z0, d_center=2.0, er=2.5, er1=1.0, twists_per_len=0.0, ptfe=False)
```

### 20.15) Rectangular Waveguide API (All Options)
Signatures:

```python
rectangular_waveguide.fc(a, b, m=1, n=0, er=1.0, mur=1.0)
rectangular_waveguide.beta(f, a, b, m=1, n=0, er=1.0, mur=1.0)
rectangular_waveguide.lambda_g(f, a, b, m=1, n=0, er=1.0, mur=1.0)
rectangular_waveguide.z_te(f, a, b, m=1, n=0, er=1.0, mur=1.0)
rectangular_waveguide.z_tm(f, a, b, m=1, n=1, er=1.0, mur=1.0)
rectangular_waveguide.a_for_fc(fc, er=1.0, mur=1.0, m=1)
rectangular_waveguide.a_for_z_te10(z0, f, er=1.0, mur=1.0)
rectangular_waveguide.length_for_angle(angle_rad, f, a, b, m=1, n=0, er=1.0, mur=1.0)
rectangular_waveguide.te10(f, a, b, er=1.0, mur=1.0)
```

Examples:

Scenario: WR-90-like waveguide calculations for cutoff, propagation, impedance, and inverse broad-wall sizing.
```python
fc10 = calc_cp.rectangular_waveguide.fc(a=22.86, b=10.16, m=1, n=0, er=1.0, mur=1.0)        # 6.55714038e9 Hz
beta = calc_cp.rectangular_waveguide.beta(f=10e9, a=22.86, b=10.16, m=1, n=0)
lg   = calc_cp.rectangular_waveguide.lambda_g(f=10e9, a=22.86, b=10.16, m=1, n=0)
zte  = calc_cp.rectangular_waveguide.z_te(f=10e9, a=22.86, b=10.16, m=1, n=0)               # 498.974376
ztm  = calc_cp.rectangular_waveguide.z_tm(f=20e9, a=22.86, b=10.16, m=1, n=1)
a_fc = calc_cp.rectangular_waveguide.a_for_fc(fc10, er=1.0, mur=1.0, m=1)                    # 22.86
a_z  = calc_cp.rectangular_waveguide.a_for_z_te10(zte, f=10e9, er=1.0, mur=1.0)              # 22.86
L90  = calc_cp.rectangular_waveguide.length_for_angle(np.pi/2, f=10e9, a=22.86, b=10.16, m=1, n=0)
te10 = calc_cp.rectangular_waveguide.te10(f=10e9, a=22.86, b=10.16)
```

### 20.16) Compatibility Alias Notes
Aliases available on `PCBCalculator`:

- `calc.coupled_microstrip` -> same object as `calc.edge_coupled_microstrip`
- `calc.coupled_stripline` -> same object as `calc.edge_coupled_stripline`

### 20.17) Argument Reference By Function
Core formula functions (`calculator.py` top-level):

- `microstrip_z0(W, th, er, t=0.0)`: `W` trace width [m], `th` substrate height [m], `er` relative permittivity, `t` conductor thickness [m].
- `microstrip_eeff(W, th, er, t=0.0)`: same geometry/medium args as `microstrip_z0`, returns quasi-static `eeff`.
- `microstrip_eeff_dispersion(W, th, er, f, t=0.0)`: adds `f` frequency [Hz] for dispersive `eeff`.
- `microstrip_z0_dispersion(W, th, er, f, t=0.0)`: adds `f` frequency [Hz] for dispersive `Z0`.
- `stripline_z0(W, b, er, t=0.0)`: `W` strip width [m], `b` ground-to-ground spacing [m], `er`, `t`.
- `coupled_stripline_zodd(W, S, b, er)`: `W` width [m], `S` edge spacing [m], `b` cavity height [m], `er`.
- `coupled_stripline_zdiff(W, S, b, er)`: same args as `coupled_stripline_zodd`, returns differential impedance.
- `broadside_stripline_zdiff_zcm(W, G, b, er)`: `W` width [m], `G` broadside spacing [m], `b` cavity height [m], `er`; returns `(Zdiff, Zcm)`.
- `cpw_z0(W, S, th, er, t=0.0, has_metal_backside=False)`: `W` center width [m], `S` slot [m], `th` substrate height [m], `er`, `t`, backside-ground flag.
- `cpw_eeff(W, S, th, er, t=0.0, has_metal_backside=False)`: same args as `cpw_z0`, returns quasi-static `eeff`.
- `cpw_eeff_dispersion(W, S, th, er, f, t=0.0, has_metal_backside=False)`: adds `f` frequency [Hz] for dispersive `eeff`.
- `cpw_z0_dispersion(W, S, th, er, f, t=0.0, has_metal_backside=False)`: adds `f` frequency [Hz] for dispersive `Z0`.
- `coax_z0(d_inner, d_outer, er)`: `d_inner` inner diameter [m], `d_outer` outer diameter [m], `er`.
- `coax_d_for_z0(Z0, d_outer, er)`: `Z0` target impedance [Ohm], `d_outer` outer diameter [m], `er`; returns inner diameter.
- `coax_cutoff_te(d_inner, d_outer, er=1.0, mur=1.0, n=1, m=1, exact=True)`: coax TE mode cutoff; `n,m` mode indices, `exact` root-solver flag.
- `coax_cutoff_tm(d_inner, d_outer, er=1.0, mur=1.0, n=0, m=1, exact=True)`: coax TM mode cutoff; `n,m` mode indices, `exact` root-solver flag.
- `twisted_pair_eeff(d_center, d_wire, er, er1=1.0, twists_per_len=0.0, ptfe=False)`: `d_center` center spacing [m], `d_wire` wire diameter [m], `er/er1` dielectric terms, `twists_per_len` turns/m, `ptfe` branch flag.
- `twisted_pair_z0(d_center, d_wire, er, er1=1.0, twists_per_len=0.0, ptfe=False)`: same args as `twisted_pair_eeff`, returns impedance.
- `twisted_pair_d_center_for_z0(z0, d_wire, er, er1=1.0, twists_per_len=0.0, ptfe=False)`: inverse center spacing from target `z0`.
- `twisted_pair_d_wire_for_z0(z0, d_center, er, er1=1.0, twists_per_len=0.0, ptfe=False)`: inverse wire diameter from target `z0`.
- `rectwg_fc(a, b, m=1, n=0, er=1.0, mur=1.0)`: `a/b` waveguide dimensions [m], `m,n` mode indices, `er/mur`.
- `rectwg_beta(f, a, b, m=1, n=0, er=1.0, mur=1.0)`: adds `f` frequency [Hz], returns propagation constant.
- `rectwg_z_te(f, a, b, m=1, n=0, er=1.0, mur=1.0)`: TE mode impedance at frequency `f`.
- `rectwg_z_tm(f, a, b, m=1, n=1, er=1.0, mur=1.0)`: TM mode impedance at frequency `f`.
- `rectwg_lambda_g(f, a, b, m=1, n=0, er=1.0, mur=1.0)`: guided wavelength.
- `rectwg_a_for_fc(fc, er=1.0, mur=1.0, m=1)`: inverse broad-wall `a` from cutoff `fc`.
- `rectwg_te10_a_for_z0(z0, f, er=1.0, mur=1.0)`: inverse TE10 broad-wall `a` from impedance and frequency.
- `coupled_microstrip_z0_even_odd(W, S, th, er, t=0.0, f=None)`: coupled-microstrip modal impedances; `W/S` geometry [m], `th` substrate height [m], `er`, `t`, optional `f`.
- `differential_cpw_zdiff_zcm(W, S_ground, S_pair, th, er, t=0.0, has_metal_backside=False, f=None)`: differential CPW/DCPWG `(Zdiff,Zcm)`; `W` width [m], `S_ground` ground slot [m], `S_pair` pair gap [m], `th`, `er`, `t`, backside flag, optional `f`.

Stackup API methods (unit-aware wrappers):

- `_MicrostripAPI.z0(w, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0)`: `w` width [unit], `layer/ground_layer` stack indices, `f0` [Hz], optional dielectric override `er`, `t` [unit].
- `_MicrostripAPI.eeff(w, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0)`: same args as `z0`, returns `eeff`.
- `_MicrostripAPI.w_for_z0(Z0, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=401)`: inverse width from target `Z0`; `w_min/w_max` bounds [unit], `n` sample count.
- `_MicrostripAPI.quarter_wave(f, layer=-1, ground_layer=0, f0=None, w=None, Z0=50.0, t=0.0)`: quarter-wave length at `f`; optional fixed `w` or auto-solved from `Z0`; `f0` for material/model evaluation.
- `_StriplineAPI.z0(w, gnd_top, gnd_bot, f0=1e9, er=None, t=0.0)`: stripline forward solve with explicit top/bottom ground indices.
- `_StriplineAPI.w_for_z0(Z0, gnd_top, gnd_bot, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=401)`: inverse strip width.
- `_EdgeCoupledStriplineAPI.zodd(w, s, gnd_top, gnd_bot, f0=1e9, er=None)`: odd-mode impedance for edge-coupled stripline.
- `_EdgeCoupledStriplineAPI.zdiff(w, s, gnd_top, gnd_bot, f0=1e9, er=None)`: differential impedance for edge-coupled stripline.
- `_EdgeCoupledStriplineAPI.w_for_zdiff(Zdiff, s, gnd_top, gnd_bot, f0=1e9, er=None, w_min=1e-6, w_max=1e-1, n=501)`: inverse width at fixed spacing.
- `_EdgeCoupledStriplineAPI.s_for_zdiff(Zdiff, w, gnd_top, gnd_bot, f0=1e9, er=None, s_min=1e-6, s_max=1e-1, n=501)`: inverse spacing at fixed width.
- `_BroadsideCoupledStriplineAPI.zdiff_zcm(w, g, gnd_top, gnd_bot, f0=1e9, er=None)`: broadside differential/common-mode impedances.
- `_BroadsideCoupledStriplineAPI.w_for_zdiff(Zdiff, g, gnd_top, gnd_bot, f0=1e9, er=None, w_min=1e-6, w_max=1e-1, n=501)`: inverse width at fixed broadside spacing.
- `_BroadsideCoupledStriplineAPI.g_for_zdiff(Zdiff, w, gnd_top, gnd_bot, f0=1e9, er=None, g_min=1e-6, g_max=1e-1, n=501)`: inverse broadside spacing at fixed width.
- `_CPWAPI.z0(w, s, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0)`: CPW/GCPW forward impedance.
- `_CPWAPI.eeff(w, s, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0)`: CPW/GCPW effective permittivity.
- `_CPWAPI.w_for_z0(Z0, s, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=501)`: inverse CPW/GCPW width.
- `_EdgeCoupledMicrostripAPI.even_odd(w, s, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0)`: coupled microstrip modal impedances `(Ze, Zo)`.
- `_EdgeCoupledMicrostripAPI.zdiff_zcm(w, s, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0)`: coupled microstrip `(Zdiff, Zcm)`.
- `_EdgeCoupledMicrostripAPI.w_for_zdiff(Zdiff, s, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=501)`: inverse width at fixed spacing.
- `_EdgeCoupledMicrostripAPI.s_for_zdiff(Zdiff, w, layer=-1, ground_layer=0, f0=1e9, er=None, t=0.0, s_min=1e-6, s_max=1e-1, n=501)`: inverse spacing at fixed width.
- `_DifferentialCPWAPI.zdiff_zcm(w, s_pair, s_ground, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0)`: differential CPW/DCPWG forward solve.
- `_DifferentialCPWAPI.w_for_zdiff(Zdiff, s_pair, s_ground, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0, w_min=1e-6, w_max=1e-1, n=501)`: inverse width at fixed pair/ground spacing.
- `_DifferentialCPWAPI.s_for_zdiff(Zdiff, w, s_ground, layer=-1, ref_layer=0, f0=1e9, er=None, t=0.0, s_min=1e-6, s_max=1e-1, n=501)`: inverse pair spacing at fixed width/ground spacing.
- `_CoaxAPI.z0(d_inner, d_outer, er=1.0)`: unit-aware coax forward impedance.
- `_CoaxAPI.d_inner_for_z0(Z0, d_outer, er=1.0)`: unit-aware coax inverse inner diameter.
- `_CoaxAPI.cutoff_te(d_inner, d_outer, er=1.0, mur=1.0, n=1, m=1, exact=True)`: unit-aware TE cutoff.
- `_CoaxAPI.cutoff_tm(d_inner, d_outer, er=1.0, mur=1.0, n=0, m=1, exact=True)`: unit-aware TM cutoff.
- `_CoaxAPI.cutoffs(d_inner, d_outer, er=1.0, mur=1.0, exact=True)`: unit-aware `(TE11, TM01)` pair.
- `_TwistedPairAPI.eeff(d_center, d_wire, er, er1=1.0, twists_per_len=0.0, ptfe=False)`: unit-aware twisted-pair `eeff`.
- `_TwistedPairAPI.z0(d_center, d_wire, er, er1=1.0, twists_per_len=0.0, ptfe=False)`: unit-aware twisted-pair impedance.
- `_TwistedPairAPI.d_center_for_z0(z0, d_wire, er, er1=1.0, twists_per_len=0.0, ptfe=False)`: unit-aware inverse center spacing.
- `_TwistedPairAPI.d_wire_for_z0(z0, d_center, er, er1=1.0, twists_per_len=0.0, ptfe=False)`: unit-aware inverse wire diameter.
- `_RectangularWaveguideAPI.fc(a, b, m=1, n=0, er=1.0, mur=1.0)`: unit-aware waveguide cutoff.
- `_RectangularWaveguideAPI.beta(f, a, b, m=1, n=0, er=1.0, mur=1.0)`: unit-aware propagation constant.
- `_RectangularWaveguideAPI.lambda_g(f, a, b, m=1, n=0, er=1.0, mur=1.0)`: unit-aware guided wavelength.
- `_RectangularWaveguideAPI.z_te(f, a, b, m=1, n=0, er=1.0, mur=1.0)`: unit-aware TE impedance.
- `_RectangularWaveguideAPI.z_tm(f, a, b, m=1, n=1, er=1.0, mur=1.0)`: unit-aware TM impedance.
- `_RectangularWaveguideAPI.a_for_fc(fc, er=1.0, mur=1.0, m=1)`: inverse `a` from cutoff.
- `_RectangularWaveguideAPI.a_for_z_te10(z0, f, er=1.0, mur=1.0)`: inverse `a` from TE10 impedance.
- `_RectangularWaveguideAPI.length_for_angle(angle_rad, f, a, b, m=1, n=0, er=1.0, mur=1.0)`: length for target phase.
- `_RectangularWaveguideAPI.te10(f, a, b, er=1.0, mur=1.0)`: convenience TE10 report dict.
- `PCBCalculator.z0(Z0, layer=-1, ground_layer=0, f0=1e9, er=None)`: backward-compatible alias to microstrip inverse width.
- `PCBCalculator.layer_index(layer)`: normalize signed layer index to absolute index.
- `PCBCalculator.z(layer)`: get layer z-coordinate in stackup units.
- `PCBCalculator.layer_distance(a, b)`: physical distance between two layers [m].
- `PCBCalculator.effective_er(layer, ground_layer, f0, er=None)`: stackup thickness-weighted dielectric extraction or direct override.

## 21) References
- [R1] E. Hammerstad, O. Jensen, "Accurate Models for Microstrip Computer-Aided Design," IEEE MTT-S Int. Microwave Symp. Digest, 1980, pp. 407-409, DOI: 10.1109/MWSYM.1980.1124303.
- [R2] S. B. Cohn, "Characteristic Impedance of the Shielded-Strip Transmission Line," IRE Trans. Microwave Theory and Techniques, vol. 2, no. 2, pp. 52-57, 1954.
- [R3] S. B. Cohn, "Shielded Coupled-Strip Transmission Line," IRE Trans. Microwave Theory and Techniques, vol. 3, no. 5, pp. 29-38, 1955, DOI: 10.1109/TMTT.1955.1124973.
- [R4] S. B. Cohn, "Characteristic Impedances of Broadside-Coupled Strip Transmission Lines," IRE Trans. Microwave Theory and Techniques, vol. 8, no. 6, pp. 633-637, 1960.
- [R5] G. Ghione, C. U. Naldi, "Coplanar Waveguides for MMIC Applications: Effect of Upper Shielding, Conductor Backing, Finite-Extent Ground Planes, and Line-to-Line Coupling," IEEE Trans. Microwave Theory and Techniques, vol. 35, no. 3, pp. 260-267, 1987, DOI: 10.1109/TMTT.1987.1133637.
- [R6] G. Ghione, C. Naldi, "Parameters of Coplanar Waveguides with Lower Ground Plane," Electronics Letters, vol. 19, pp. 734-735, 1983.
- [R7] M. Kirschning, R. H. Jansen, "Accurate Model for Effective Dielectric Constant of Microstrip with Validity up to Millimetre-Wave Frequencies," Electronics Letters, vol. 18, no. 6, pp. 272-273, 1982.
- [R8] M. Kirschning, R. H. Jansen, "Accurate Wide-Range Design Equations for the Frequency-Dependent Characteristic of Coupled Microstrip Lines," IEEE Trans. Microwave Theory and Techniques, vol. 32, no. 1, pp. 83-90, 1984.
- [R9] Qucs Technical Documentation, "Twisted pair" (quasi-static model with `acosh` and `q` correction). Online documentation mirror: `https://qucs.github.io/tech/node93.html`.
- [R10] D. M. Pozar, Microwave Engineering, 4th ed., Wiley, 2011. (Standard coax, two-wire line, and rectangular-waveguide relations.)
- [R11] M. Abramowitz, I. A. Stegun (eds.), Handbook of Mathematical Functions, NBS, 1964. (Elliptic integrals and related numerical forms.)
- [R12] T. C. Edwards, M. B. Steer, Foundations for Microstrip Circuit Design, 4th ed., Wiley, 2016. (Broadside and coupled stripline equation summaries used for cross-checking.)
- [R13] Qucs Transcalc implementation source (coplanar/coupled formulations): `qucs-transcalc/coplanar.cpp` in the Qucs repository (`https://github.com/Qucs/qucs`).
- [R14] Qucs technical/transcalc formula lineage used as implementation cross-check reference for several quasi-static models (Qucs docs + transcalc sources).
- [R15] Qucs Transcalc `microstrip.cpp` (dispersion model implementation used for `eeff(f)` and `Z0(f)` coefficient sets).
- [R16] Qucs Transcalc `c_microstrip.cpp` (coupled-microstrip static and frequency-dependent even/odd model implementation).
- [R17] Qucs Transcalc `stripline.cpp` (finite-thickness centered-stripline impedance model implementation).

## 22) Change Log (Recent)
- Added exact coax modal cutoff solver (annular-waveguide TE/TM eigen-root bisection) with approximation fallback.
- Added microstrip frequency-dispersion formulas (`eeff(f)`, `Z0(f)`) and wired stackup APIs to use `f0`.
- Added edge-coupled microstrip frequency-dependent even/odd impedance model (Kirschning/Jansen coefficient sets).
- Added CPW/GCPW dispersion support and propagated it into differential CPW/DCPWG capacitance decomposition.
- Added stripline finite-thickness impedance branch and exposed `t` in stripline API forward/reverse calls.
- Added twisted-pair inverse geometry solvers (`d_center_for_z0`, `d_wire_for_z0`).
- Expanded validation coverage with dispersion/inverse/thickness checks and refreshed CSV outputs (`70/70` pass), including EMerge-native calculator consistency tests.
- Added inverse refinement (bracketed bisection on top of sampled interpolation).
- Added differential CPW/DCPWG forward and reverse APIs in both codebases.
- Added broadside-coupled stripline forward/reverse APIs in both codebases.
- Upgraded broadside implementation from simplified closed form to full Cohn implicit-k formulation.
- Added worked forward/inverse examples for every implemented topology API.
- Expanded examples into a full usage guide with layer/material setup patterns, full method signatures, and all optional parameters.
- Added per-function argument reference in markdown and inline function headers in `calculator.py` for public formula/API callables.
- Fixed EMerge inverse-solver vectorized evaluation for edge-coupled stripline and broadside `g_for_zdiff`.
- Added EMerge-native unit tests for edge-coupled stripline and broadside stripline inverse round-trips.
