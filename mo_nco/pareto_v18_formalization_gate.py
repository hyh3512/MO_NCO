"""Fail-closed Lean4 formalization status gate.

A source file containing no literal ``sorry`` is not a machine proof.  This
module distinguishes source scanning, compilation, axiom-closure inspection,
and paper-statement fidelity.  In a toolchain-free environment the only valid
status is ``NOT_PERFORMED_TOOLCHAIN_MISSING``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess


_FORBIDDEN = re.compile(r"\b(sorry|admit|axiom)\b")


@dataclass(frozen=True)
class FormalizationGateResult:
    lean: str | None
    lake: str | None
    source_file_count: int
    source_forbidden_tokens: tuple[str, ...]
    compiled: bool
    axiom_closure_checked: bool
    fidelity_checked: bool
    status: str
    log: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lean": self.lean,
            "lake": self.lake,
            "source_file_count": self.source_file_count,
            "source_forbidden_tokens": list(self.source_forbidden_tokens),
            "compiled": self.compiled,
            "axiom_closure_checked": self.axiom_closure_checked,
            "fidelity_checked": self.fidelity_checked,
            "status": self.status,
            "log": self.log,
        }


def _scan_sources(formal_root: Path) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    sources = tuple(sorted(formal_root.rglob("*.lean")))
    findings: list[str] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        # Remove line comments before the conservative token scan.  Block
        # comments are not stripped; a false positive fails closed.
        stripped = "\n".join(line.split("--", 1)[0] for line in text.splitlines())
        for match in _FORBIDDEN.finditer(stripped):
            findings.append(f"{source.relative_to(formal_root)}:{match.group(1)}")
    return sources, tuple(findings)


def run_formalization_gate(
    project_root: Path,
    *,
    require_compile: bool = False,
    exported_theorems: tuple[str, ...] = (),
) -> FormalizationGateResult:
    lean = shutil.which("lean")
    lake = shutil.which("lake")
    sources, findings = _scan_sources(project_root / "formal")
    if not sources:
        raise RuntimeError("no Lean source files were found")
    if lean is None or lake is None:
        result = FormalizationGateResult(
            lean=lean,
            lake=lake,
            source_file_count=len(sources),
            source_forbidden_tokens=findings,
            compiled=False,
            axiom_closure_checked=False,
            fidelity_checked=False,
            status="NOT_PERFORMED_TOOLCHAIN_MISSING",
            log="Lean4/lake was not found; source text and Python tests are not compilation evidence.",
        )
        if require_compile:
            raise RuntimeError(result.log)
        return result

    completed = subprocess.run(
        [lake, "build"],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0 or findings:
        result = FormalizationGateResult(
            lean=lean,
            lake=lake,
            source_file_count=len(sources),
            source_forbidden_tokens=findings,
            compiled=False,
            axiom_closure_checked=False,
            fidelity_checked=False,
            status="COMPILE_OR_SOURCE_CLOSURE_FAILED",
            log=completed.stdout,
        )
        if require_compile:
            raise RuntimeError(completed.stdout)
        return result

    # Axiom closure requires an explicit theorem list.  Merely compiling the
    # project is not enough because an axiom may be hidden in a dependency.
    if not exported_theorems:
        status = "COMPILED_AXIOM_CLOSURE_NOT_CHECKED"
        axiom_checked = False
        axiom_log = "No exported theorem list was supplied for #print axioms."
    else:
        audit_source = project_root / "formal" / "V18AxiomAudit.lean"
        audit_source.write_text(
            "\n".join(["import ParetoSMCV18DeterministicCore", *[f"#print axioms {name}" for name in exported_theorems]])
            + "\n",
            encoding="utf-8",
        )
        audit = subprocess.run(
            [lake, "env", "lean", str(audit_source)],
            cwd=project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        axiom_log = audit.stdout
        axiom_checked = audit.returncode == 0 and "sorryAx" not in audit.stdout
        status = (
            "COMPILED_AXIOM_CLOSURE_CHECKED_FIDELITY_PENDING"
            if axiom_checked
            else "AXIOM_CLOSURE_FAILED"
        )

    result = FormalizationGateResult(
        lean=lean,
        lake=lake,
        source_file_count=len(sources),
        source_forbidden_tokens=findings,
        compiled=True,
        axiom_closure_checked=axiom_checked,
        fidelity_checked=False,
        status=status,
        log=completed.stdout + "\n" + axiom_log,
    )
    if require_compile and not (result.compiled and result.axiom_closure_checked):
        raise RuntimeError(result.log)
    return result


__all__ = ["FormalizationGateResult", "run_formalization_gate"]
