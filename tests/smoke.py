"""Offline tiny Transformers plus hand-checkable finite-domain/evaluation cases."""

import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    BertConfig,
    BertModel,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)
from scoperag.cli import train, evaluate, canonical
from scoperag.domain import version
from scoperag.llm import LocalModel
from scoperag.method import infer
from scoperag.model import ApplicabilityModel, Heads
from scoperag.utils import load, save, seed_all


def tiny_models(output):
    """Create local random fixtures; no pretrained weights or remote calls."""
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[EOS]",
        "[BOS]",
        "0",
        "1",
        "2",
        "feature",
        "supports",
        "cpu",
        "gpu",
        "platform",
        "yes",
        "no",
        "unknown",
    ]
    tokenizer = Tokenizer(
        WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        eos_token="[EOS]",
        bos_token="[BOS]",
    )
    fast.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    encoder, causal = output / "tiny_encoder", output / "tiny_causal"
    config = BertConfig(
        vocab_size=len(words),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    BertModel(config).save_pretrained(encoder)
    fast.save_pretrained(encoder)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(words),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=512,
            bos_token_id=5,
            eos_token_id=4,
            pad_token_id=0,
        )
    ).save_pretrained(causal)
    fast.save_pretrained(causal)
    return encoder, causal


def main():
    torch.set_num_threads(1)
    seed_all(17)
    out = Path("outputs/smoke").resolve()
    out.mkdir(parents=True, exist_ok=True)
    encoder, causal = tiny_models(out)
    annotations = []
    for i in range(10):
        annotations.append(
            dict(
                query=["feature cpu", "feature gpu", "feature platform unspecified"][
                    i % 3
                ],
                fact="feature supports cpu",
                qualifier="platform cpu",
                slot="latency" if i % 2 == 0 else "feature",
                source_span="feature supports cpu; platform cpu",
                product_family="train_toy" if i < 8 else "development_toy",
                split="train" if i < 8 else "development",
                data_kind="synthetic_functional",
                applicability=["entailed", "inconsistent", "unknown"][i % 3],
                slot_support=i % 2,
            )
        )
    save(out / "annotations.json", annotations)
    cfg = load("configs/default.json")
    save(out / "config.json", dict(cfg, epochs=1, batch_size=4))
    seed_all(17)
    initial = ApplicabilityModel(str(encoder)).state_dict()
    train(
        argparse.Namespace(
            config=str(out / "config.json"),
            annotations=str(out / "annotations.json"),
            seed=17,
            encoder=str(encoder),
            revision="main",
            device="cpu",
            output=str(out / "heads"),
        )
    )
    state = torch.load(out / "heads/heads.pt", weights_only=True)
    for name in [
        "applicability.weight",
        "support.weight",
        "encoder.embeddings.word_embeddings.weight",
    ]:
        assert torch.isfinite(state[name]).all() and not torch.equal(
            initial[name], state[name]
        ), name
    metadata = load(out / "heads/model.json")
    metadata["encoder"] = "unavailable-original-model"
    save(out / "heads/model.json", metadata)
    heads, copy = Heads(out / "heads", "cpu"), Heads(out / "heads", "cpu")
    tokens = heads.tokenizer(
        ["feature cpu", "feature gpu"], return_tensors="pt", padding=True
    )
    with torch.no_grad():
        a, b = heads.model(tokens), copy.model(tokens)
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])
    heads.model.train()
    logits, support = heads.model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor([0, 1])
    ) + torch.nn.functional.binary_cross_entropy_with_logits(
        support, torch.tensor([1.0, 0.0])
    )
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for n, p in heads.model.named_parameters()
        if n
        in [
            "applicability.weight",
            "support.weight",
            "encoder.embeddings.word_embeddings.weight",
        ]
    )
    llm = LocalModel(str(causal), max_generated=8, max_total=256, device="cpu")
    llm.chat("feature", "smoke", 2, temperature=0)
    probabilities = llm.option_probs("feature", 2)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-6)
    assert llm.ledger and all(r["prompt_tokens"] > 0 for r in llm.ledger)
    llm.reset()
    llm.max_generated = 0
    try:
        llm.chat("feature", "budget_check", 1)
        raise AssertionError("Exhausted generation budget must fail")
    except RuntimeError as error:
        assert "budget" in str(error).lower()
    query = dict(
        question="three slots",
        scope={},
        slots=["x", "y", "z"],
        domain=dict(
            version=["1.0"],
            platform=["cpu"],
            hardware=["one"],
            backend=["A", "B", "C"],
            settings=["one"],
        ),
    )
    sets = [["A", "B"], ["B", "C"], ["A", "C"], ["B"]]
    facts = [
        dict(
            id=str(i),
            assertion=f"slot {i}",
            qualifier="scoped",
            source_span=f"slot {i} scoped",
            url="https://example.org/fixture",
            scope={"allowed": {"backend": allowed}},
            parsed=True,
            residual=[],
            values={query["slots"][min(i, 2)]: "yes"},
        )
        for i, allowed in enumerate(sets)
    ]
    scores = [
        dict(
            semantic_probabilities=[1.0, 0.0, 0.0],
            slot_scores=[
                float(j == min(i, 2)) * (0.9 if i == 3 else 1.0) for j in range(3)
            ],
        )
        for i in range(4)
    ]
    local_cfg = dict(cfg, max_facts=3, max_bundles=1)
    full, _, domain = infer(
        query, facts, scores, local_cfg, "scoperag", lambda s: len(s.split())
    )
    individual, _, _ = infer(
        query, facts, scores, local_cfg, "individual", lambda s: len(s.split())
    )
    exact, _, _ = infer(
        query, facts, scores, local_cfg, "exact_atomic", lambda s: len(s.split())
    )
    assert full["status"] == "conditional" and individual["status"] == "clarification"
    assert (
        canonical(full) == canonical(exact) and "3" in full["branches"][0]["fact_ids"]
    )
    assert version("1.10") > version("1.9")
    q = domain.mask({})
    assert domain.applicability(q, q) == "entailed"
    assert (
        domain.applicability(q, domain.mask({"allowed": {"backend": ["A"]}}))
        == "unknown"
    )
    assert domain.applicability(q, np.zeros_like(q)) == "inconsistent"
    opaque = deepcopy(facts)
    for fact in opaque:
        fact["scope"]["allowed"]["driver"] = ["unparsed"]
    answer, _, _ = infer(query, opaque, scores, local_cfg, "scoperag", len)
    assert answer["status"] == "clarification"
    limited, _, _ = infer(
        query, facts, scores, dict(local_cfg, evidence_tokens=1), "scoperag", len
    )
    assert limited["status"] == "clarification"
    label = dict(
        id="a",
        structured_answer={"status": "clarification", "branches": []},
        pair_id="pair",
        pair_stratum="joint",
    )
    save(out / "labels.json", [label, dict(label, id="b")])
    rows = [
        dict(
            id=i,
            method="scoperag",
            seed=17,
            dataset="toy",
            top_k=20,
            query=query,
            ledger=[],
            answer={"status": "clarification", "branches": [], "claims": []},
            **({"error": "deliberate fixture error"} if i == "a" else {}),
        )
        for i in ["a", "b"]
    ]
    save(out / "traces.json", rows)
    evaluate(
        argparse.Namespace(
            labels=str(out / "labels.json"),
            judgments=None,
            judge_model=None,
            inputs=[str(out / "traces.json")],
            output=str(out / "metrics.json"),
        )
    )
    metrics = load(out / "metrics.json")["summary"][0]
    assert (
        metrics["answer_accuracy_percent"] == 50.0
        and metrics["paired_accuracy_percent"] == 0.0
    )
    assert (
        metrics["runtime_error_count"] == 1
        and metrics["pair_strata"]["joint"]["source_pairs"] == 1
    )
    print(
        "PASS: offline tiny encoder 2-step fitting and gradients, portable checkpoint, actual causal forward/generation/token caps, finite-domain joint/exact/individual controls, unknown restrictions and hand pair/error evaluation"
    )


if __name__ == "__main__":
    main()
