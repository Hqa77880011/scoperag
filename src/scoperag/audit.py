import json
from pathlib import Path

from .domain import Domain
from .utils import load, save


def judge_citations(args):
    from .llm import LocalModel

    facts = {r["id"]: r for r in load(Path(args.index) / "index.json")["facts"]}
    llm = LocalModel(args.model, args.revision, max_generated=4096, max_total=65536)
    judgments = []
    for trace in load(args.traces):
        llm.reset()
        domain = Domain(trace["query"]["domain"])
        answer = trace["answer"]
        branches = {b["id"]: b for b in answer.get("branches", [])}
        supported, valid_links, completeness = [], [], []
        for claim in answer.get("claims", []):
            branch = branches.get(claim.get("branch_id"))
            citations = [facts[i] for i in claim.get("fact_ids", []) if i in facts]
            response = llm.json(
                "Independently evaluate the emitted statement against the original quoted source spans. A claim lacking a necessary qualifier is unsupported. Return {claim_supported:boolean,link_supported:[boolean for each supplied citation]}. Judge each citation as well as their joint support.\nCLAIM="
                + json.dumps(claim)
                + "\nBRANCH="
                + json.dumps(branch)
                + "\nSOURCES="
                + json.dumps(citations),
                "sealed_citation_audit",
                512,
            )
            links = response["link_supported"]
            if type(response.get("claim_supported")) is not bool:
                raise ValueError("Citation judge claim_supported must be a boolean")
            if len(links) != len(citations) or any(type(x) is not bool for x in links):
                raise ValueError("Citation judge must return one boolean per citation")
            local = []
            for fact, support in zip(citations, links):
                cells = (
                    domain.mask({"relation": branch["conditions"]}) if branch else None
                )
                valid = bool(
                    branch
                    and support
                    and fact.get("parsed", False)
                    and all(~cells | domain.mask(fact["scope"]))
                    and set(fact.get("residual", []))
                    <= set(trace["query"].get("known_prerequisites", []))
                )
                local.append(valid)
            local.extend([False] * (len(claim.get("fact_ids", [])) - len(citations)))
            valid_links.extend(local)
            supported.append(bool(response["claim_supported"] and any(local)))
            completeness.append(any(local))
        judgments.append(
            {
                "id": trace["id"],
                "method": trace["method"],
                "seed": trace["seed"],
                "claim_supported": supported,
                "citation_valid": valid_links,
                "claim_has_valid_citation": completeness,
                "auditor_model": args.model,
                "inference": llm.ledger.copy(),
            }
        )
        save(args.output, judgments)
