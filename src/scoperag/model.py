import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class ApplicabilityModel(nn.Module):
    def __init__(self, name="microsoft/deberta-v3-base", revision="main"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name, revision=revision)
        self.applicability = nn.Linear(self.encoder.config.hidden_size, 3)
        self.support = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, tokens):
        hidden = self.encoder(**tokens).last_hidden_state[:, 0]
        return self.applicability(hidden), self.support(hidden).squeeze(-1)


class Heads:
    def __init__(self, directory, device):
        from .utils import load

        config = load(str(directory) + "/model.json")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["encoder"], revision=config["revision"]
        )
        self.model = ApplicabilityModel(config["encoder"], config["revision"]).to(
            device
        )
        self.model.load_state_dict(
            torch.load(
                str(directory) + "/heads.pt", map_location=device, weights_only=True
            )
        )
        self.model.eval()
        self.device = device

    def score(self, query, facts, slots):
        result = []
        for fact in facts:
            left = [query + "\nSLOT: " + slot for slot in slots]
            right = [
                fact["assertion"] + "\nQUALIFIERS: " + fact.get("qualifier", "")
            ] * len(slots)
            encoded = self.tokenizer(
                left,
                right,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                applicability, support = self.model(encoded)
            probabilities = applicability.softmax(-1).mean(0).cpu().numpy()
            result.append(
                {
                    "semantic_probabilities": probabilities.tolist(),
                    "slot_scores": support.sigmoid().cpu().numpy().tolist(),
                }
            )
        return result
