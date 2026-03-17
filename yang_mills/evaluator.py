"""
Evaluator for Yang-Mills Existence and Mass Gap — Hybrid Mode.

Two-stage evaluation:
  Stage 1 (Script): Lean 4 type-checker — does the proof compile? How many sorry-free lemmas?
  Stage 2 (Critic): Adversarial mathematical review — is the approach sound? Any gaps?

The critic score is provided externally via --metrics flag (from the critic agent).
The script score is computed here.

Scoring:
  - lean_compilation: 1.0 if compiles, 0.0 if not (weight: 0.25)
  - sorry_ratio: sorry-free / total declarations (weight: 0.15)
  - milestone_score: fraction of 10 key milestones (weight: 0.10)
  - proof_depth: how deep toward the actual proof (not just scaffolding) (weight: 0.15)
  - mathematical_rigor: from critic (weight: 0.20)
  - novelty_of_approach: from critic (weight: 0.15)
  - combined_score = weighted sum

Milestones (10 key results needed):
  M1: Define SU(N) gauge group with correct mathematical structure
  M2: Define principal G-bundle or connection on R^4
  M3: Define curvature / field strength tensor
  M4: Define Yang-Mills action functional (not placeholder)
  M5: State Yang-Mills equations as PDE
  M6: Prove energy non-negativity (real proof, not placeholder)
  M7: Define quantum Hilbert space with operator algebra
  M8: Formalize Wightman or Osterwalder-Schrader axioms (non-trivial)
  M9: Define mass gap using spectral theory
  M10: Any non-trivial theorem reducing the main problem

Proof Depth Scoring (0.0-1.0):
  0.0: Only sorry stubs
  0.1: Type-correct definitions (placeholders like := 0 or := True)
  0.2: Mathematically meaningful definitions (actual gauge theory objects)
  0.3: Non-trivial auxiliary lemmas proved
  0.4: Key structural results (e.g., gauge invariance properties)
  0.5: Partial results toward main theorem (e.g., 2D or 3D Yang-Mills)
  0.6: Significant sub-problems solved (e.g., lattice theory formalized)
  0.7: Major components of a proof strategy implemented
  0.8: Complete proof strategy with few remaining gaps
  0.9: All gaps filled except final assembly
  1.0: Complete proof of the Millennium Prize Problem
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import time


MILESTONES = {
    "M1_gauge_group": r"(SU|SpecialUnitaryGroup|GaugeGroup).*(:=|where)",
    "M2_principal_bundle": r"(PrincipalBundle|Connection|GaugeConnection).*(:=|where)",
    "M3_curvature": r"(Curvature|FieldStrength|CurvatureForm).*(:=|where)",
    "M4_ym_action": r"(YangMillsAction|YMAction|yangMillsFunctional).*(:=|where)",
    "M5_ym_equations": r"(YangMillsEquation|yangMillsEq|ymEquation).*(:=|where)",
    "M6_energy_nonneg": r"(energy_nonneg|energyNonneg|ymEnergyPos)",
    "M7_hilbert_space": r"(YMHilbert|quantumHilbert|FockSpace).*(:=|where)",
    "M8_wightman": r"(WightmanAxiom|OsterwalderSchrader|axiom_)",
    "M9_mass_gap": r"(MassGap|massGap|spectralGap).*(:=|where)",
    "M10_nontrivial": r"theorem\s+\w+.*YangMills|theorem\s+\w+.*ym_|theorem\s+\w+.*gauge",
}

PLACEHOLDER_PATTERNS = [
    r':=\s*0\b',           # := 0
    r':=\s*True\b',        # := True
    r'fun\s+_\w*\s*=>',    # fun _A =>
    r':=\s*trivial\b',     # := trivial
    r':=\s*rfl\b',         # := rfl (for type equalities, not real content)
]

DEEP_MATH_PATTERNS = [
    (r'(Sobolev|sobolev|H1|H2)', 0.05, "Sobolev space usage"),
    (r'(gauge_transform|gaugeInvariant|GaugeEquiv)', 0.05, "Gauge invariance"),
    (r'(lattice|Lattice.*gauge|discretize)', 0.05, "Lattice gauge theory"),
    (r'(renormali[sz]|Renorm)', 0.05, "Renormalization"),
    (r'(continuum_limit|continuumLimit)', 0.05, "Continuum limit"),
    (r'(functional_integral|pathIntegral|FunctionalIntegral)', 0.05, "Path integral"),
    (r'(asymptotic_freedom|asymptoticFreedom)', 0.05, "Asymptotic freedom"),
    (r'(confinement|Confinement)', 0.05, "Confinement"),
    (r'(Wilson.*loop|wilsonLoop)', 0.05, "Wilson loops"),
    (r'(Osterwalder.*Schrader|reflection_positivity)', 0.05, "OS axioms"),
]


def count_sorries(content):
    decl_pattern = r'(?:theorem|lemma|def|instance|noncomputable def)\s+\w+'
    all_decls = re.findall(decl_pattern, content)
    total = len(all_decls)
    sorry_count = content.count('sorry')
    sorry_free = max(0, total - sorry_count)
    return sorry_free, total


def check_milestones(content):
    achieved = {}
    for name, pattern in MILESTONES.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            region = content[match.start():match.start() + 500]
            has_sorry = 'sorry' in region
            achieved[name] = not has_sorry
        else:
            achieved[name] = False
    return achieved


def compute_proof_depth(content):
    """Estimate how deep the proof goes beyond scaffolding."""
    score = 0.0
    sorry_free, total = count_sorries(content)

    if total == 0:
        return 0.0

    # Base: has any declarations at all
    if total > 0:
        score = 0.05

    # Has sorry-free definitions
    if sorry_free > 0:
        score = 0.1

    # Check for placeholder vs real definitions
    placeholder_count = sum(len(re.findall(p, content)) for p in PLACEHOLDER_PATTERNS)
    real_def_ratio = max(0, 1.0 - placeholder_count / max(total, 1))
    if real_def_ratio > 0.3:
        score = max(score, 0.2)

    # Check for deep mathematical content
    deep_score = 0.0
    for pattern, value, _name in DEEP_MATH_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            deep_score += value
    score = max(score, 0.2 + min(deep_score, 0.3))

    # Check for substantive proofs (not just `trivial`, `rfl`, `norm_num`)
    proof_tactics = re.findall(r'by\s*\n\s*((?:(?!theorem|lemma|def|instance)[\s\S])*?)(?=\n(?:theorem|lemma|def|instance|end|--|/-|\Z))', content)
    substantive_proofs = 0
    for proof in proof_tactics:
        trivial_tactics = ['trivial', 'rfl', 'norm_num', 'simp', 'exact']
        lines = [l.strip() for l in proof.strip().split('\n') if l.strip()]
        if len(lines) > 2 and not all(any(t in l for t in trivial_tactics) for l in lines):
            substantive_proofs += 1

    if substantive_proofs >= 3:
        score = max(score, 0.4)
    if substantive_proofs >= 5:
        score = max(score, 0.5)

    # Check sorry count — fewer sorries = deeper
    if total > 0:
        sorry_ratio = sorry_free / total
        if sorry_ratio > 0.95 and total > 20:
            score = max(score, 0.3 + 0.3 * sorry_ratio)

    return min(score, 1.0)


def run_lean_check(candidate_path, project_dir):
    try:
        dest = os.path.join(project_dir, 'YangMills', 'Candidate.lean')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(candidate_path, 'r') as f:
            content = f.read()
        with open(dest, 'w') as f:
            f.write(content)

        env = os.environ.copy()
        env['PATH'] = os.path.expanduser('~/.elan/bin') + ':' + env.get('PATH', '')

        result = subprocess.run(
            ['lake', 'env', 'lean', dest],
            capture_output=True, text=True, timeout=180,
            cwd=project_dir, env=env
        )
        compiles = result.returncode == 0
        errors = result.stderr if result.stderr else ""
        return compiles, errors, content
    except subprocess.TimeoutExpired:
        return False, "Lean check timed out after 180s", ""
    except Exception as e:
        return False, f"Failed to run lean: {e}", ""


def evaluate(candidate_path):
    start_time = time.time()
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = eval_dir

    try:
        with open(candidate_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return {"combined_score": 0.0, "error": f"Cannot read: {e}"}

    if not content.strip().startswith(('import', 'open', 'namespace', 'section',
                                       'theorem', 'lemma', 'def', 'structure',
                                       'class', 'noncomputable', '/-', '--', '#')):
        return {"combined_score": 0.0, "error": "Not a valid Lean file"}

    # Stage 1: Lean compilation
    compiles, errors, _ = run_lean_check(candidate_path, project_dir)
    lean_compilation = 1.0 if compiles else 0.0

    # Stage 2: Sorry ratio
    sorry_free, total_decls = count_sorries(content)
    sorry_ratio = sorry_free / max(total_decls, 1)

    # Stage 3: Milestones
    milestones = check_milestones(content)
    achieved = sum(1 for v in milestones.values() if v)
    milestone_score = achieved / len(MILESTONES)

    # Stage 4: Proof depth
    proof_depth = compute_proof_depth(content)

    # Combined score (script portion only — critic adds to this)
    # Weights: lean=0.25, sorry=0.15, milestone=0.10, depth=0.15, critic_rigor=0.20, critic_novelty=0.15
    script_score = (0.25 * lean_compilation +
                    0.15 * sorry_ratio +
                    0.10 * milestone_score +
                    0.15 * proof_depth)
    # Critic scores (mathematical_rigor, novelty_of_approach) will be added externally
    # For script-only mode, scale up: combined = script_score / 0.65
    combined = min(script_score / 0.65, 1.0)

    eval_time = time.time() - start_time

    result = {
        "combined_score": round(combined, 12),
        "lean_compilation": lean_compilation,
        "sorry_ratio": round(sorry_ratio, 6),
        "sorry_free": sorry_free,
        "total_declarations": total_decls,
        "milestone_score": round(milestone_score, 6),
        "milestones_achieved": {k: v for k, v in milestones.items() if v},
        "milestones_missing": [k for k, v in milestones.items() if not v],
        "proof_depth": round(proof_depth, 4),
        "eval_time": round(eval_time, 3),
    }

    if not compiles and errors:
        result["compilation_errors"] = errors[:3000]

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluator.py <candidate.lean>")
        sys.exit(1)
    result = evaluate(sys.argv[1])
    print(json.dumps(result, indent=2))
