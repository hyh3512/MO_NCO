from __future__ import annotations

"""Executable cell-separated adaptive pilot matching the v16 theorem."""

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Callable, Sequence

from .pareto_adaptive_type_cell import anytime_hoeffding_radius_upper
from .pareto_independent_replica_certificate import (
    canonical_rational_string,
    clopper_pearson_lower_bracket,
    parse_canonical_probability,
)

ADAPTIVE_PILOT_RESULT_SCHEMA_V16 = "pareto_adaptive_cell_separated_pilot_v16"

class AdaptivePilotError(ValueError):
    pass

@dataclass(frozen=True)
class AdaptivePilotCellResult:
    cell_id: str
    selected_type: str
    elimination_winner: str
    stopping_round: int
    empirical_means: tuple[tuple[str, str], ...]
    cp_lower_bounds: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class AdaptivePilotResult:
    schema: str
    type_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    familywise_identification_error: str
    familywise_cp_error: str
    total_replica_evaluations: int
    cell_results: tuple[AdaptivePilotCellResult, ...]
    selection_rule: str
    probability_scope: str
    def to_jsonable(self) -> dict[str, object]:
        payload=asdict(self); payload["cell_results"]=[asdict(x) for x in self.cell_results]; return payload

def run_cell_separated_successive_elimination(*, type_ids: Sequence[str], cell_ids: Sequence[str],
                                               sample: Callable[[str,str,int],bool],
                                               familywise_identification_error: Fraction | str,
                                               familywise_cp_error: Fraction | str,
                                               max_rounds: int) -> AdaptivePilotResult:
    """Run a cell-separated balanced successive-elimination pilot.

    The callback must use an independent stream for every (type, cell, index)
    triple.  This is a theorem assumption; the function does not infer it from
    a PRNG implementation.
    """
    types=tuple(sorted(type_ids)); cells=tuple(sorted(cell_ids))
    if len(types)<2 or not cells or len(set(types))!=len(types) or len(set(cells))!=len(cells):
        raise AdaptivePilotError("Need unique IDs, at least two types, and one cell.")
    alpha_id=parse_canonical_probability(familywise_identification_error,label="familywise_identification_error")
    alpha_cp=parse_canonical_probability(familywise_cp_error,label="familywise_cp_error")
    if not (0<alpha_id<1 and 0<alpha_cp<1): raise AdaptivePilotError("Errors must lie in (0,1).")
    if isinstance(max_rounds,bool) or not isinstance(max_rounds,int) or max_rounds<=0:
        raise AdaptivePilotError("max_rounds must be positive.")
    active={cell:set(types) for cell in cells}; stopped={}
    successes={(t,c):0 for c in cells for t in types}; counts={(t,c):0 for c in cells for t in types}
    total=0
    for n in range(1,max_rounds+1):
        for cell in cells:
            if cell in stopped: continue
            for type_id in sorted(active[cell]):
                obs=sample(type_id,cell,counts[(type_id,cell)])
                if not isinstance(obs,bool): raise AdaptivePilotError("sample must return bool.")
                counts[(type_id,cell)]+=1; successes[(type_id,cell)]+=int(obs); total+=1
            radius=anytime_hoeffding_radius_upper(n,type_count=len(types),cell_count=len(cells),familywise_error=alpha_id)
            means={t:Fraction(successes[(t,cell)],counts[(t,cell)]) for t in active[cell]}
            max_lower=max(means[t]-radius for t in active[cell])
            active[cell].difference_update({t for t in active[cell] if means[t]+radius<max_lower})
            if len(active[cell])==1: stopped[cell]=n
        if len(stopped)==len(cells): break
    if len(stopped)!=len(cells): raise AdaptivePilotError("Pilot did not resolve all cells.")
    alpha_pair=alpha_cp/(len(types)*len(cells)); results=[]
    for cell in cells:
        brackets={t:clopper_pearson_lower_bracket(successes[(t,cell)],counts[(t,cell)],alpha_pair) for t in types}
        elimination_winner=next(iter(active[cell]))
        selected=min(types,key=lambda t:(-brackets[t].lower,t))
        results.append(AdaptivePilotCellResult(
            cell_id=cell,selected_type=selected,elimination_winner=elimination_winner,
            stopping_round=stopped[cell],
            empirical_means=tuple((t,canonical_rational_string(Fraction(successes[(t,cell)],counts[(t,cell)]))) for t in types),
            cp_lower_bounds=tuple((t,canonical_rational_string(brackets[t].lower)) for t in types),
        ))
    return AdaptivePilotResult(
        schema=ADAPTIVE_PILOT_RESULT_SCHEMA_V16,type_ids=types,cell_ids=cells,
        familywise_identification_error=canonical_rational_string(alpha_id),
        familywise_cp_error=canonical_rational_string(alpha_cp),
        total_replica_evaluations=total,cell_results=tuple(results),
        selection_rule="successive_elimination_then_simultaneous_CP_max",
        probability_scope="ideal_independent_cell_separated_Bernoulli_streams",
    )

__all__=["ADAPTIVE_PILOT_RESULT_SCHEMA_V16","AdaptivePilotCellResult","AdaptivePilotError",
         "AdaptivePilotResult","run_cell_separated_successive_elimination"]
