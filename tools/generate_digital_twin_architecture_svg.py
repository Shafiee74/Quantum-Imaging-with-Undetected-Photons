#!/usr/bin/env python3
"""Generate the code-grounded QIUP digital-twin architecture overview.

The layout is deliberately deterministic and dependency free.  Connectors are
drawn before opaque block cards, use dedicated routing gutters, and carry no
free-floating labels.  Every documented DT block therefore owns its visible
name, input, and output text, while the exhaustive connection semantics remain
in DIGITAL_TWIN_BLOCK_CONNECTIONS.md.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "evidence"
    / "digital_twin_system_architecture"
    / "DIGITAL_TWIN_ARCHITECTURE.svg"
)
IO_DOCUMENT = OUTPUT.with_name("DIGITAL_TWIN_BLOCK_IO_PHYSICS.md")

PAGE_W = 4300
PAGE_H = 2620
CARD_W = 550
CARD_H = 360


CATEGORIES = {
    "control": ("#E8EDF2", "#506675", "#D3DDE4"),
    "physical": ("#DCEEFF", "#1F4E79", "#B9DAF5"),
    "spatial": ("#DDF4F5", "#227C83", "#BDE5E7"),
    "detector": ("#DFF4E4", "#2A7F62", "#BFE6CA"),
    "data": ("#EAE6FA", "#65528F", "#D5CCF1"),
    "evidence": ("#FFF0CF", "#A46712", "#F4DCA8"),
    "external": ("#F4E7FB", "#7A5195", "#E4CEF0"),
}


CARDS = {
    "EXT-USER": (50, 180, "external", "User / automation",
        "Defines the scientific case, target object, acquisition, and requested product.",
        "External: object transmission T(x,y), phase phi(x,y), profile A/B/C/D/C_lab, seed.",
        "[DT-45, DT-01] Physical request, object/mask, acquisition and evidence command.",
        "Object intensity 0<=T<=1; phase [rad]; frame count; ideal/noisy selection."),
    "DT-45": (430, 180, "external", "Interactive interface and bridge",
        "Translates controls into a simulation request and streams previews and results.",
        "[EXT-USER] Profile/object/noise controls; [DT-30] scientific result and statistics.",
        "[DT-01] Structured request; [EXT-OUT] preview, image arrays, run summary.",
        "Selected profile; mask/image; seed; detector-noise and display controls."),
    "DT-01": (810, 180, "control", "Execution entry and request dispatch",
        "Dispatches one run, a frame sequence, dataset generation, replay, or evidence task.",
        "[EXT-USER, DT-45] Request; [DT-33] manifest-resolved replay request.",
        "[DT-02] Setup selection; [DT-03] physical state; [DT-31, DT-43] job controls.",
        "Reference object; profile/YAML; random seed; frames; sweep; noise switch."),
    "DT-02": (1190, 180, "control", "Physical configuration resolution",
        "Resolves A/B/C/D/C_lab, validates domains, and fixes optical architecture/geometry.",
        "[DT-01] Profile or configuration file and explicit physical overrides.",
        "[DT-04] Valid setup; [DT-11, DT-13, DT-14, DT-21, DT-22] optical/acquisition parameters.",
        "Wavelengths; crystal/poling; path/lens geometry; object plane; detector settings."),
    "DT-03": (1570, 180, "control", "Physical variations and calibration state",
        "Maps engineering variations to effective pump, alignment, thermal, focus, SLM and detector state.",
        "[DT-01] Preset or measured variation; [DT-02] allowed ranges and couplings.",
        "[DT-04, DT-10, DT-12, DT-16, DT-17, DT-21] Effective perturbations and validity.",
        "Pump stability; gain request; alignment; temperature; detector; cooling; focus; SLM."),
    "DT-04": (1950, 180, "control", "Fail-closed physical preflight",
        "Rejects inconsistent energy, geometry, gain, detector, object, or unsupported-model requests before RNG use.",
        "[DT-01] Reference object; [DT-02] setup; [DT-03] effective physical state.",
        "[DT-10, DT-12, DT-20] Approved request; [DT-30] support/validity classification.",
        "Energy conservation; array shape; canonical hashes; parameter bounds; forbidden states."),

    "DT-11": (50, 400, "physical", "Crystal, QPM, dispersion and thermal physics",
        "Solves refractive index, quasi-phase matching, optical-path phase/coherence, thermal focus and magnification.",
        "[DT-02] Pump/signal/idler wavelengths; ppKTP length L, poling period Lambda(T), temperature, paths/lenses.",
        "[DT-10] Phase mismatch Delta k/nonlinearity; [DT-14, DT-16, DT-17] mapping, phase, coherence, focus.",
        "k_j=2pi n_j/lambda_j; Delta k=k_p-k_s-k_i-2pi m/Lambda(T); OPD; L_coh; phi_OPD."),
    "DT-10": (430, 400, "physical", "Pump transport and SPDC operating point",
        "Computes local pump power for both interactions and exact squeezing/mean modal occupation.",
        "[DT-04] Approved source request; [DT-03] pump variation; [DT-11] Delta k, d_eff, L and focusing.",
        "[DT-14] pump generation weight; [DT-15] nbar_1, nbar_2 and gain regime; [DT-30] source audit.",
        "P_p1,P_p2 [mW]; r_0=k_SPDC sqrt(P_local); nbar=sinh^2(r) with detuned exact branches."),
    "DT-12": (810, 400, "spatial", "Object and spatial-light-modulator response",
        "Builds the physical complex object response and applies SLM quantization, leakage, gamma and registration.",
        "[DT-04] Valid object intensity/phase maps; [DT-03] SLM calibration and misregistration state.",
        "[DT-13] Complex amplitude transmission t_o(x,y); [DT-33] object evidence/hash.",
        "t_o=sqrt(T_o) exp(i phi_o); T_o dimensionless; phi_o [rad]; pixel pitch [um]."),
    "DT-13": (1190, 400, "spatial", "Architecture-specific optical transfer",
        "Applies the actual two-crystal or one-crystal double-pass object/idler/signal path.",
        "[DT-12] Complex object field; [DT-02] A/B/C/D topology, transforms, losses and mirror behavior.",
        "[DT-14] Returned/mapped idler field; [DT-15] idler seed a_i and signal return a_s.",
        "Two crystal: one object pass. Single transmissive: outbound*return. Reflective: object replaces mirror."),
    "DT-14": (1570, 400, "spatial", "SPDC spatial correlation and camera mapping",
        "Maps object to signal-camera coordinates and applies SPDC correlation separately from ordinary optical blur.",
        "[DT-13] Returned object field; [DT-11] wavelengths/geometry; [DT-10] pump waist and generation weight.",
        "[DT-15] Effective idler seed/signal marginal; [DT-30, DT-41] widths, magnification, orientation.",
        "Near field: joint Gaussian. Far field: q_s=-q_i. C_lab: pump spectrum times sinc^2 phase matching."),
    "DT-20": (1950, 400, "detector", "Independent random streams and fixed patterns",
        "Partitions one seed into repeatable optics/detector streams and fixes PRNU/hot maps across a sequence.",
        "[DT-04] Approved seed/shape; [DT-03] detector and cooling variation levels.",
        "[DT-17] optics-noise stream; [DT-21] plus/minus streams, PRNU and hot-pixel maps.",
        "Master seed; frame index; pixel response variation; hot-pixel probability; independent port draws."),

    "DT-15": (810, 650, "physical", "Exact SPDC source moments",
        "Forms first-order signal moments for independent two-crystal sources or the two-squeezer Michelson path.",
        "[DT-10] nbar_1,nbar_2; [DT-13, DT-14] idler seed a_i and signal return a_s; signal losses.",
        "[DT-16] mean signal occupations N_1,N_2 and mutual coherence C_12; [DT-41] reference observables.",
        "g^(1)=|C_12|/sqrt(N_1 N_2); V=2|C_12|/(N_1+N_2); exact at low/high modal gain."),
    "DT-16": (1190, 650, "physical", "Induced coherence and signal combiner",
        "Applies path/spectral coherence and phase, then combines the two signal fields at complementary ports.",
        "[DT-15] N_1,N_2,C_12; [DT-11] OPD/coherence; [DT-03] phase/alignment; spectral filter.",
        "[DT-17] I_plus,I_minus,D_ideal; [DT-30, DT-41] ideal visibility and coherence evidence.",
        "I_+/-=(N_1+N_2)/2 +/- Re[C_12 exp(i(theta+Delta phi))]. Idler is not detected."),
    "DT-17": (1570, 650, "physical", "Signal-image formation and optical distortions",
        "Applies collection, diffraction/defocus/aberration PSF, affine mapping and optional etalon modulation after interference.",
        "[DT-16] Ideal signal ports; [DT-11, DT-03] wavelength, aperture, focus/wavefront/alignment; [DT-20] optics RNG.",
        "[DT-21] Signal photons per pixel per exposure; [DT-22] detector-bypass clean ports; [DT-30] optical audit.",
        "sigma_opt^2=sigma_diff^2+sigma_defocus^2+sigma_aberr^2. Config C uses Delta z_eff once."),
    "DT-21": (1950, 650, "detector", "EMCCD or photon-counting detector",
        "Converts incident signal photons to analog ADU or binary photon-counted frames in physical stage order.",
        "[DT-17] I_plus/I_minus photons; [DT-20] RNG/PRNU/hot maps; [DT-02, DT-03] detector/cooling state.",
        "[DT-22] Measured plus/minus frames and detector audit; [DT-30] detector metadata.",
        "mu_e=QE*PRNU*(I+background)+dark*time+CIC; Poisson -> EM Gamma gain -> read noise -> ADC."),
    "DT-22": (2330, 650, "detector", "Frame acquisition and phase reconstruction",
        "Repeats optical/detector frames, selects the signal port, forms differences and optionally fits phase steps.",
        "[DT-21] Measured ports; [DT-17] clean ports; [DT-02] frames, port and phase schedule theta_k.",
        "[DT-30] Plus/minus/difference stacks, selected image, clean difference, phase/modulation/SNR maps.",
        "Least squares: I(theta)=a+b cos(theta)+c sin(theta); recovered phase phi=atan2(-c,b)."),

    "DT-30": (810, 940, "data", "In-memory scientific result",
        "Packages images, stacks, selected port, physical state, support status and execution evidence.",
        "[DT-22] Acquisition; [DT-02, DT-03, DT-10..DT-17, DT-21] resolved parameters/audits.",
        "[DT-31, DT-33, DT-41, DT-42, DT-43, DT-45] Arrays, metadata, validity and evidence.",
        "Ideal and measured signal images; plus/minus/difference stacks; phase maps; source/object/array hashes."),
    "DT-31": (1190, 940, "data", "Synthetic-dataset generation",
        "Validates sweeps, samples allowed physical states, runs the shared forward model and indexes successes.",
        "[DT-01, DT-02, DT-03] Reference, setup, sweep/sampling policy; [DT-30] one result per sample.",
        "[DT-32] Candidate arrays/metadata; [DT-33] dataset index and configuration evidence.",
        "Sweep quantities; sample count; base seed; frames/sample; constrained physical-state distribution."),
    "DT-32": (1570, 940, "data", "Validated atomic sample commit",
        "Writes a temporary sample, reloads/verifies it, then atomically commits only complete finite data.",
        "[DT-31] Sample identity and candidate arrays; [DT-30] result metadata and required shapes.",
        "[DT-31] Committed sample path/index; [DT-33] file hashes and integrity status.",
        "Required arrays; dimensions; finite values; metadata schema; SHA-256; atomic rename."),
    "DT-33": (1950, 940, "data", "Provenance, integrity and replay",
        "Hashes source/setup/object/arrays, reconstructs runs from manifests, and detects replay mismatch.",
        "[DT-30] Result/execution state; [DT-32] committed files; [DT-02, DT-03] resolved setup/state.",
        "[DT-01] Replay request/verdict; [DT-43, DT-44] integrity evidence; [EXT-OUT] manifest.",
        "Canonical JSON; source/config/object/array SHA-256; environment identity; numerical replay comparison."),
    "EXT-OUT": (2330, 940, "external", "Scientific products",
        "Collects the user-facing simulation, dataset, verification, validation and publication artifacts.",
        "[DT-33] Provenance; [DT-43] figures/reports; [DT-44] release verdict; [DT-45] UI products.",
        "External: scientific images, synthetic datasets, tables, reports and reproducibility package.",
        "PNG/PDF/SVG; NumPy arrays; CSV/JSON; metadata; checksums; explicit Simulation/Experimental labels."),

    "DT-40": (50, 1220, "evidence", "Canonical benchmark targets",
        "Defines common transparent/blocked/point/slit/edge/bar/amplitude/phase truth for matched A-D tests.",
        "External: target type, physical grid/pixel pitch, profile set, quiet state and seed.",
        "[DT-30] Reference objects for production runs; [DT-41] identical ground truth and coordinates.",
        "Object transmission/phase; feature size [um]; field of view [um]; profile-independent target identity."),
    "DT-41": (430, 1220, "evidence", "Independent physics and numerical verification",
        "Compares production results against independently implemented equations, convergence and parity studies.",
        "[DT-40] Truth targets; [DT-30] results; [DT-11, DT-15, DT-16] physical laws; [DT-46] paths.",
        "[DT-43] Metrics/tables/gates; [DT-44] regression/parity result.",
        "Absolute/relative error; visibility; FWHM; energy balance; convergence rate; Monte Carlo confidence."),
    "EXT-LAB": (810, 1220, "external", "Read-only experimental archive",
        "Supplies measured apparatus data without relabeling it as simulated or altering source files.",
        "External: TIFF frames, displacement/position records, apparatus metadata and calibration partition.",
        "[DT-42] Experimental images/curves and frozen calibration/held-out membership.",
        "Camera counts; physical positions [um/mm]; effective axial displacement Delta z_eff; acquisition metadata."),
    "DT-42": (1190, 1220, "evidence", "Configuration-C apparatus validation",
        "Calibrates only declared partitions and evaluates held-out longitudinal/transverse experimental observables.",
        "[DT-30] C_lab prediction; [EXT-LAB] experimental data; [DT-46] data-root/output policy.",
        "[DT-43] Simulation-versus-Experimental curves/images, calibration records and held-out metrics.",
        "Visibility, displacement Delta z_eff, transverse position, fitted/calibrated quantities and residuals."),
    "DT-43": (1570, 1220, "evidence", "Evidence and publication figures",
        "Generates figures directly from saved simulation/verification/experimental sources with explicit provenance.",
        "[DT-30, DT-33] Simulation/data lineage; [DT-41] verification; [DT-42] experiment; [DT-46] layout/style.",
        "[DT-44] Release artifacts; [EXT-OUT] figures, tables, reports and reproduction instructions.",
        "Scientific units/labels; Simulation versus Experimental identity; fixed color/style; source checksums."),
    "DT-44": (1950, 1220, "evidence", "Build, regression and release integrity",
        "Builds the single-file distribution, checks modular parity/tests, and verifies package-wide checksums.",
        "Source modules; [DT-30] parity case; [DT-41] tests; [DT-43] evidence and figures.",
        "[DT-43, EXT-OUT] Release verdict, parity/regression reports, build hashes and SHA-256 manifest.",
        "Bit-exact parity; test counts; authoritative module hashes; complete-file manifest coverage."),
    "DT-46": (2330, 1220, "control", "Evidence output and data-root policy",
        "Creates safe timestamped layouts and resolves external experimental roots without modifying source data.",
        "Repository root, task identity, artifact type and optional read-only experimental-data root.",
        "[DT-41, DT-42, DT-43] Canonical data/figure/table/log/config/provenance/report paths.",
        "Task stamp; directory role; artifact name; external-root provenance; no physical calculation."),
}


EDGES = [
    # Request/configuration path.
    ("runtime", [(380, 252), (430, 252)]),
    ("runtime", [(760, 252), (810, 252)]),
    ("runtime", [(1140, 252), (1190, 252)]),
    ("runtime", [(1900, 252), (1950, 252)]),
    ("interface", [(215, 324), (215, 332), (900, 332), (900, 324)]),
    ("control", [(975, 180), (975, 150), (1735, 150), (1735, 180)]),
    ("control", [(1355, 180), (1355, 138), (2115, 138), (2115, 180)]),
    # Configuration/preflight fan-out through the clear inter-lane gutter.
    ("parameter", [(1355, 324), (1355, 342), (215, 342), (215, 400)]),
    ("parameter", [(1735, 324), (1735, 352), (595, 352), (595, 400)]),
    ("parameter", [(1735, 324), (1735, 362), (975, 362), (975, 400)]),
    ("control", [(2115, 324), (2115, 372), (595, 372), (595, 400)]),
    ("control", [(2115, 324), (2115, 382), (975, 382), (975, 400)]),
    ("control", [(2115, 324), (2115, 400)]),
    # Physical/spatial path.
    ("parameter", [(380, 472), (430, 472)]),
    ("runtime", [(1140, 472), (1190, 472)]),
    ("runtime", [(1520, 472), (1570, 472)]),
    ("runtime", [(595, 544), (595, 610), (900, 610), (900, 650)]),
    ("runtime", [(1355, 544), (1355, 620), (975, 620), (975, 650)]),
    ("runtime", [(1735, 544), (1735, 600), (1050, 600), (1050, 650)]),
    ("parameter", [(215, 544), (215, 575), (1275, 575), (1275, 650)]),
    ("runtime", [(1140, 722), (1190, 722)]),
    ("runtime", [(1520, 722), (1570, 722)]),
    ("runtime", [(1900, 722), (1950, 722)]),
    ("runtime", [(2280, 722), (2330, 722)]),
    ("parameter", [(2115, 544), (2115, 650)]),
    ("parameter", [(1950, 472), (1915, 472), (1915, 625), (1810, 625), (1810, 650)]),
    ("parameter", [(1735, 794), (1735, 842), (2495, 842), (2495, 794)]),
    # Acquisition into products and dataset/provenance path.
    ("runtime", [(2495, 794), (2495, 875), (975, 875), (975, 940)]),
    ("runtime", [(1140, 1012), (1190, 1012)]),
    ("runtime", [(1520, 1012), (1570, 1012)]),
    ("runtime", [(1900, 1012), (1950, 1012)]),
    ("evidence", [(2280, 1012), (2330, 1012)]),
    # Result stream to UI, routed only through the left inter-card gutter.
    ("interface", [(810, 1012), (785, 1012), (785, 340), (595, 340), (595, 324)]),
    # Replay loop around the outer right/top gutter.
    ("interface", [(2280, 1030), (2735, 1030), (2735, 112), (1050, 112), (1050, 180)]),
    # Verification, validation, evidence, and release.
    ("evidence", [(380, 1292), (430, 1292)]),
    ("evidence", [(1140, 1292), (1190, 1292)]),
    ("evidence", [(1520, 1292), (1570, 1292)]),
    ("evidence", [(1900, 1292), (1950, 1292)]),
    ("evidence", [(975, 1084), (975, 1152), (595, 1152), (595, 1220)]),
    ("evidence", [(1035, 1084), (1035, 1164), (1355, 1164), (1355, 1220)]),
    ("evidence", [(760, 1292), (785, 1292), (785, 1400), (1735, 1400), (1735, 1364)]),
    ("parameter", [(2495, 1220), (2495, 1180), (1735, 1180), (1735, 1220)]),
    ("parameter", [(2415, 1220), (2415, 1168), (595, 1168), (595, 1220)]),
    ("parameter", [(2455, 1220), (2455, 1174), (1355, 1174), (1355, 1220)]),
    ("evidence", [(1735, 1220), (1735, 1146), (2495, 1146), (2495, 1084)]),
    ("evidence", [(2115, 1220), (2115, 1134), (2555, 1134), (2555, 1084)]),
]


def scale_x(value: int) -> int:
    """Map the approved compact seven-column geometry to wider detail cards."""
    if value < 50:
        return value
    for index in range(7):
        old_left = 50 + 380 * index
        new_left = 50 + 600 * index
        if old_left <= value <= old_left + 330:
            return round(new_left + (value - old_left) * CARD_W / 330)
        if old_left + 330 < value < old_left + 380:
            return round(new_left + CARD_W + (value - old_left - 330))
    return round(4200 + (value - 2660) * 100 / 140)


Y_ANCHORS = [
    (0, 0), (112, 110), (125, 125), (138, 145), (150, 160),
    (180, 180), (252, 360), (324, 540), (330, 548), (342, 562),
    (350, 570), (382, 612), (400, 620), (472, 800), (544, 980),
    (575, 1000), (600, 1016), (610, 1024), (620, 1032), (625, 1036),
    (650, 1040), (722, 1220), (794, 1400), (842, 1435), (850, 1440),
    (875, 1445), (890, 1450), (940, 1500), (1012, 1680), (1030, 1725),
    (1084, 1860), (1125, 1875), (1134, 1880), (1146, 1886),
    (1152, 1890), (1164, 1895), (1168, 1897), (1170, 1900),
    (1174, 1905), (1180, 1912), (1220, 1980), (1292, 2160),
    (1364, 2340), (1400, 2350),
]


def scale_y(value: int) -> int:
    for (old_a, new_a), (old_b, new_b) in zip(Y_ANCHORS, Y_ANCHORS[1:]):
        if old_a <= value <= old_b:
            fraction = (value - old_a) / (old_b - old_a)
            return round(new_a + fraction * (new_b - new_a))
    return round(value * PAGE_H / 1710)


def wrapped_lines(value: str, width: int = 65, maximum: int = 3) -> list[str]:
    return textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False)[:maximum]


def card_svg(block_id: str, spec: tuple[int, int, str, str, str, str, str, str]) -> str:
    old_x, old_y, category, title, action, inputs, outputs, quantities = spec
    x, y = scale_x(old_x), scale_y(old_y)
    fill, stroke, header = CATEGORIES[category]
    title_lines = wrapped_lines(title, 52, 2)
    action_lines = wrapped_lines(action, 65, 3)
    in_lines = wrapped_lines(inputs, 65, 3)
    out_lines = wrapped_lines(outputs, 65, 3)
    quantity_lines = wrapped_lines(quantities, 65, 4)
    quantity_label = "PHYSICS" if category in {"physical", "spatial", "detector"} else "PARAMETERS"
    parts = [
        f'<g id="{escape(block_id)}" transform="translate({x} {y})">',
        f'<rect width="{CARD_W}" height="{CARD_H}" rx="13" class="card" fill="{fill}" stroke="{stroke}"/>',
        f'<path d="M13 0 H537 Q550 0 550 13 V52 H0 V13 Q0 0 13 0 Z" fill="{header}"/>',
        f'<text x="16" y="23" class="blockId">{escape(block_id)}</text>',
    ]
    if len(title_lines) == 1:
        parts.append(f'<text x="90" y="24" class="blockTitle">{escape(title_lines[0])}</text>')
    else:
        parts.append(f'<text x="90" y="18" class="blockTitle smallTitle">{escape(title_lines[0])}</text>')
        parts.append(f'<text x="16" y="40" class="blockTitle smallTitle">{escape(title_lines[1])}</text>')

    fields = [
        ("ACTION", action_lines, 78, 124),
        ("INPUT", in_lines, 148, 194),
        ("OUTPUT", out_lines, 218, 264),
        (quantity_label, quantity_lines, 288, None),
    ]
    for label, lines, y0, divider_y in fields:
        parts.append(f'<text x="16" y="{y0}" class="ioLabel">{label}</text>')
        for n, line in enumerate(lines):
            parts.append(f'<text x="98" y="{y0 + n * 17}" class="ioText">{escape(line)}</text>')
        if divider_y is not None:
            parts.append(f'<line x1="16" y1="{divider_y}" x2="534" y2="{divider_y}" class="divider"/>')
    parts.append('</g>')
    return "\n".join(parts)


def edge_svg(kind: str, points: list[tuple[int, int]]) -> str:
    coords = " ".join(f"{scale_x(x)},{scale_y(y)}" for x, y in points)
    styles = {
        "runtime": ('#244B64', '2.7', '', 'arrowRuntime'),
        "control": ('#506675', '2.3', '8 6', 'arrowRuntime'),
        "parameter": ('#738794', '2.0', '7 6', 'arrowParameter'),
        "evidence": ('#A46712', '2.5', '', 'arrowEvidence'),
        "interface": ('#7A5195', '2.1', '8 6', 'arrowInterface'),
    }
    stroke, width, dash, marker = styles[kind]
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    return (
        f'<polyline points="{coords}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'
        f'{dash_attr} marker-end="url(#{marker})"/>'
    )


def legend_box(x: int, category: str, label: str) -> str:
    fill, stroke, _ = CATEGORIES[category]
    return (
        f'<g transform="translate({x} 2420)">'
        f'<rect width="350" height="46" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        f'<text x="175" y="29" text-anchor="middle" class="legendText">{escape(label)}</text>'
        '</g>'
    )


def build_svg() -> str:
    lane_specs = [
        (30, 125, 4240, 425, "REQUEST, CONFIGURATION, AND GOVERNANCE"),
        (30, 570, 4240, 850, "PHYSICAL / COMPUTATIONAL FORWARD MODEL"),
        (30, 1450, 4240, 410, "RESULTS, DATASET, AND PROVENANCE"),
        (30, 1900, 4240, 440, "VERIFICATION, VALIDATION, EVIDENCE, AND RELEASE"),
    ]
    lanes = []
    for x, y, w, h, label in lane_specs:
        lanes.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
            'fill="#FFFFFF" stroke="#D5E0E7" stroke-width="1.5"/>'
        )
        lanes.append(
            f'<g transform="translate({x + 20} {y + 14})"><rect width="600" height="34" rx="8" fill="#FFFFFF" stroke="#D5E0E7" stroke-width="1"/>'
            f'<text x="16" y="23" class="laneLabel">{escape(label)}</text></g>'
        )

    paths = "\n".join(edge_svg(kind, points) for kind, points in EDGES)
    cards = "\n".join(card_svg(block_id, spec) for block_id, spec in CARDS.items())
    legends = "\n".join(
        [
            legend_box(50, "control", "Control / configuration"),
            legend_box(420, "physical", "SPDC / optical physics"),
            legend_box(790, "spatial", "Object / spatial mapping"),
            legend_box(1160, "detector", "Detector / acquisition"),
            legend_box(1530, "data", "Results / provenance"),
            legend_box(1900, "evidence", "Verification / evidence"),
            legend_box(2270, "external", "External / interface"),
        ]
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_W}" height="{PAGE_H}" viewBox="0 0 {PAGE_W} {PAGE_H}" role="img" aria-labelledby="title desc">
  <title id="title">QIUP Digital Twin System Architecture with explicit block inputs and outputs</title>
  <desc id="desc">Code-grounded block diagram. Each functional block contains its own inputs and outputs. Orthogonal arrows are routed behind opaque cards and never cover block text. Colors identify functional roles.</desc>
  <defs>
    <filter id="shadow" x="-15%" y="-15%" width="130%" height="135%"><feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#18324A" flood-opacity="0.13"/></filter>
    <marker id="arrowRuntime" markerWidth="11" markerHeight="11" refX="9" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,8 L10,4 z" fill="#244B64"/></marker>
    <marker id="arrowParameter" markerWidth="11" markerHeight="11" refX="9" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,8 L10,4 z" fill="#738794"/></marker>
    <marker id="arrowEvidence" markerWidth="11" markerHeight="11" refX="9" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,8 L10,4 z" fill="#A46712"/></marker>
    <marker id="arrowInterface" markerWidth="11" markerHeight="11" refX="9" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,8 L10,4 z" fill="#7A5195"/></marker>
    <style>
      text {{ font-family: Arial, Helvetica, sans-serif; fill: #173247; }}
      .pageTitle {{ font-size: 36px; font-weight: 700; }}
      .pageSub {{ font-size: 16px; fill: #526777; }}
      .lane {{ fill: #FFFFFF; fill-opacity: .58; stroke: #D5E0E7; stroke-width: 1.5; }}
      .laneLabelBox {{ fill: #FFFFFF; stroke: #D5E0E7; stroke-width: 1; }}
      .laneLabel {{ font-size: 15px; font-weight: 700; letter-spacing: .55px; fill: #506675; }}
      .card {{ stroke-width: 1.8; }}
      .blockId {{ font-size: 15px; font-weight: 800; }}
      .blockTitle {{ font-size: 15px; font-weight: 700; }}
      .smallTitle {{ font-size: 13.5px; }}
      .ioLabel {{ font-size: 11.5px; font-weight: 800; fill: #334E60; }}
      .ioText {{ font-size: 11.7px; fill: #334E60; }}
      .divider {{ stroke: #718795; stroke-opacity: .28; stroke-width: 1; }}
      .edge {{ fill: none; stroke-linejoin: round; stroke-linecap: round; }}
      .runtime {{ stroke: #244B64; stroke-width: 2.7; marker-end: url(#arrowRuntime); }}
      .control {{ stroke: #506675; stroke-width: 2.3; stroke-dasharray: 8 6; marker-end: url(#arrowRuntime); }}
      .parameter {{ stroke: #738794; stroke-width: 2.0; stroke-dasharray: 7 6; marker-end: url(#arrowParameter); }}
      .evidence {{ stroke: #A46712; stroke-width: 2.5; marker-end: url(#arrowEvidence); }}
      .interface {{ stroke: #7A5195; stroke-width: 2.1; stroke-dasharray: 8 6; marker-end: url(#arrowInterface); }}
      .legendText {{ font-size: 13px; font-weight: 700; }}
      .noteText {{ font-size: 13px; fill: #526777; }}
    </style>
  </defs>
  <rect width="{PAGE_W}" height="{PAGE_H}" fill="#F6F9FB"/>
  <text x="50" y="55" class="pageTitle">QIUP Digital Twin - Code-Grounded System Architecture</text>
  <text x="50" y="85" class="pageSub">Each block states its action, source-connected physical inputs, destination-connected outputs, and real scientific quantities.</text>
  {''.join(lanes)}
  <g id="connections">{paths}</g>
  <g id="blocks">{cards}</g>
  <g id="legend">{legends}
    <g transform="translate(2650 2402)"><rect width="1600" height="100" rx="12" fill="#FFFFFF" stroke="#CCD9E1" stroke-width="1.4"/>
      <text x="20" y="27" class="noteText">Solid navy: runtime data/physical flow | dashed gray: control or parameter dependency</text>
      <text x="20" y="52" class="noteText">Amber: verification/evidence flow | dashed purple: interface or replay feedback</text>
      <text x="20" y="77" class="noteText">[DT-xx] inside INPUT/OUTPUT identifies the connected source/destination block; all connector lines remain behind cards.</text>
    </g>
  </g>
  <text x="50" y="2555" class="pageSub">Definitions, units, equations and assumptions: DIGITAL_TWIN_BLOCK_IO_PHYSICS.md. Exhaustive connection IDs: DIGITAL_TWIN_BLOCK_CONNECTIONS.md. Signal is detected; idler is not directly detected.</text>
</svg>
'''


def build_io_markdown() -> str:
    """Create the semantic companion to the SVG from the same block records."""
    lines = [
        "# QIUP Digital Twin Block I/O and Physical Quantities",
        "",
        "This document is generated with `DIGITAL_TWIN_ARCHITECTURE.svg` from",
        "`tools/generate_digital_twin_architecture_svg.py`. Names below are physical or",
        "scientific names; implementation symbol names remain in",
        "`DIGITAL_TWIN_CODE_TRACEABILITY.md`.",
        "",
        "## Reading the diagram",
        "",
        "- `[DT-xx]` in an input identifies the source block.",
        "- `[DT-xx]` in an output identifies the destination/consumer block.",
        "- Physical quantities include symbols and units where the implementation has a physical unit.",
        "- Dimensionless quantities are explicitly described as dimensionless or normalized.",
        "- Signal photons are detected. The idler field is not directly detected; it changes signal coherence.",
        "",
        "## Block actions, connected inputs, connected outputs, and parameters",
        "",
    ]
    for block_id, spec in CARDS.items():
        _, _, category, title, action, inputs, outputs, quantities = spec
        lines.extend(
            [
                f"### {block_id} - {title}",
                "",
                f"- **Functional role:** {category}.",
                f"- **Action:** {action}",
                f"- **Inputs and source blocks:** {inputs}",
                f"- **Outputs and destination blocks:** {outputs}",
                f"- **Physical quantities / parameters:** {quantities}",
                "",
            ]
        )
    lines.extend(
        [
            "## Physical symbol definitions and units",
            "",
            "| Symbol | Physical meaning | Unit / domain | Primary owner |",
            "|---|---|---|---|",
            "| `lambda_p`, `lambda_s`, `lambda_i` | Pump, detected-signal, and undetected-idler vacuum wavelengths | nm in setup; converted consistently for equations | DT-11 |",
            "| `n_p`, `n_s`, `n_i` | Temperature-dependent ppKTP refractive indices | dimensionless | DT-11 |",
            "| `L` | Nonlinear-crystal interaction length | mm in setup; converted to m or um as required | DT-11/DT-10 |",
            "| `Lambda(T)` | Temperature-dependent poling period | um | DT-11 |",
            "| `m` | Quasi-phase-matching order | integer | DT-11 |",
            "| `Delta k` | Longitudinal phase mismatch `k_p-k_s-k_i-2*pi*m/Lambda(T)` | inverse length | DT-11 |",
            "| `d_eff` | Effective nonlinear coefficient including duty-cycle/QPM factor | pm/V metadata | DT-11/DT-10 |",
            "| `P_p1`, `P_p2` | Local pump powers at SPDC interactions 1 and 2 | mW | DT-10 |",
            "| `w_p` | Pump waist controlling generation and momentum correlation | um | DT-10/DT-14 |",
            "| `r`, `r_0` | Dimensionless squeezing amplitude | dimensionless | DT-10 |",
            "| `nbar_1`, `nbar_2` | Mean photon occupation per modeled SPDC mode | photons/mode, dimensionless expectation | DT-10 |",
            "| `T_o(x,y)` | Object intensity transmission | dimensionless, 0 to 1 | DT-12 |",
            "| `phi_o(x,y)` | Object phase delay | rad | DT-12 |",
            "| `t_o(x,y)` | Complex object amplitude `sqrt(T_o) exp(i phi_o)` | dimensionless complex amplitude | DT-12/DT-13 |",
            "| `a_i`, `t_seed` | Complex returned idler/seed amplitude after object, mode overlap, and idler-path loss | dimensionless complex amplitude | DT-13/DT-14 |",
            "| `a_s` | Complex returned signal amplitude for the Michelson second interaction | dimensionless complex amplitude | DT-13 |",
            "| `N_1`, `N_2` | Mean signal photon occupations reaching the two combiner inputs | photons/mode | DT-15 |",
            "| `C_12` | Complex first-order mutual coherence between signal alternatives | same occupation scale as `N_j` | DT-15 |",
            "| `g^(1)` | Normalized first-order coherence `|C_12|/sqrt(N_1 N_2)` | dimensionless, 0 to 1 physically | DT-15 |",
            "| `V` | Ideal 50:50 visibility `2|C_12|/(N_1+N_2)` | dimensionless | DT-15/DT-16 |",
            "| `OPD` | Signed optical-path difference between signal alternatives | mm internally where declared | DT-11/DT-16 |",
            "| `L_coh` | Signal coherence length associated with the spectral filter | same length unit as OPD | DT-11/DT-16 |",
            "| `phi_OPD` | Carrier phase `2*pi*OPD/lambda_s` | rad | DT-11/DT-16 |",
            "| `theta` | Controlled combiner/phase-step phase | rad | DT-16/DT-22 |",
            "| `I_plus`, `I_minus` | Complementary ideal detected-signal port intensities | photons/pixel/exposure after image formation | DT-16/DT-17 |",
            "| `D_ideal` | Ideal complementary-port difference `I_plus-I_minus` | same as intensity | DT-16 |",
            "| `sigma_NF`, `sigma_FF` | Near-field and far-field SPDC correlation widths | physical camera/object coordinate, with explicit pixel conversion | DT-14 |",
            "| `q_s`, `q_i` | Signal and idler transverse spatial frequencies | inverse length | DT-14 |",
            "| `M` | Signed idler-object to signal-camera magnification/orientation | dimensionless | DT-11/DT-14 |",
            "| `sigma_opt` | Ordinary post-interference optical PSF width | physical/pixel width after explicit conversion | DT-17 |",
            "| `Delta z_eff` | Effective axial displacement for Configuration C double-pass geometry | um or mm as labeled; double-pass conversion applied once | DT-17/DT-42 |",
            "| `QE` | Detector quantum efficiency | electron/photon probability, dimensionless | DT-21 |",
            "| `PRNU` | Pixel-response nonuniformity multiplier | dimensionless map | DT-20/DT-21 |",
            "| `dark_rate` | Thermally generated electron rate | electrons/pixel/s | DT-21 |",
            "| `CIC` | Clock-induced charge | electrons/pixel/frame | DT-21 |",
            "| `G` | Electron-multiplication gain | output electrons/input electron | DT-21 |",
            "| `sigma_read` | Post-EM read-noise standard deviation | electrons | DT-21 |",
            "| `ADU` | Digitized camera output | analog-to-digital units | DT-21/DT-22 |",
            "",
            "## Implemented physical equations and interpretation",
            "",
            "1. **Energy conservation:** `1/lambda_p = 1/lambda_s + 1/lambda_i` within configured tolerance.",
            "2. **QPM:** `Delta k = k_p-k_s-k_i-2*pi*m/Lambda(T)`, with `k_j=2*pi*n_j/lambda_j`.",
            "3. **Calibrated gain:** `r_0=k_SPDC*sqrt(P_local)` and `nbar=sinh^2(r_0)` at phase match; detuned hyperbolic, oscillatory, and critical branches are evaluated exactly.",
            "4. **Object field:** `t_o=sqrt(T_o)*exp(i*phi_o)`; object loss, mode overlap, path loss, and pump generation weight remain separate factors.",
            "5. **Moments:** A/B use two-crystal independent-source idler seeding; C/D use the one-crystal forward/backward two-squeezer Michelson moments. Low/high gain use the same exact formulas.",
            "6. **Combiner:** `I_plus/minus=(N_1+N_2)/2 +/- Re[C_12 exp(i(theta+Delta phi))]`.",
            "7. **Near field:** conditional joint-Gaussian position correlation. **Far field:** momentum anticorrelation `q_s=-q_i`; C_lab additionally integrates pump angular spectrum times sinc-squared phase matching over idler momentum.",
            "8. **Thermal coherence:** `V_coh=exp[-(|OPD|/L_coh)^2]`; `phi_OPD=2*pi*OPD/lambda_s`.",
            "9. **Optical PSF:** `sigma_opt^2=sigma_diffraction^2+sigma_defocus^2+sigma_aberration^2`, applied after interference and kept separate from SPDC correlation blur.",
            "10. **Detector mean:** `mu_e=QE*PRNU*(I_signal+I_background)+dark_rate*exposure+CIC`; Poisson input electrons, hot charge, EM Gamma gain, saturation, read noise, bias and ADC follow in that order.",
            "",
            "## Scope and non-claims",
            "",
            "- The current forward model is a reduced-order object-conditioned QIUP model, not a full four-dimensional biphoton-wavefunction propagation.",
            "- High-gain modal moments are implemented; high-gain spatial prediction for C/D remains provisional.",
            "- The analog EMCCD is the manuscript path. Photon counting is explicit opt-in compatibility/test behavior.",
            "- Vignetting is disabled; Zernike wavefront and measured two-dimensional pump-profile requests fail preflight.",
            "- A clean simulated image is a detector-bypass result from the same model, not experimental ground truth.",
            "- Simulation and Experimental data remain explicitly labeled and are never interchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_svg(), encoding="utf-8", newline="\n")
    IO_DOCUMENT.write_text(build_io_markdown(), encoding="utf-8", newline="\n")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {IO_DOCUMENT}")
    print(f"Blocks: {sum(1 for key in CARDS if key.startswith('DT-'))}; external interfaces: {sum(1 for key in CARDS if key.startswith('EXT-'))}; connectors: {len(EDGES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
