import gzip
import io
import pickle
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

from .utils import load, save


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if tag in ("p", "li", "h1", "h2", "h3", "pre"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class DataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(
            "CONFACT input must contain plain data without executable objects"
        )


def fetch(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "ScopeRAG-implementation/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def download_documents(registry, output):
    documents = []
    for row in load(registry):
        parser = TextParser()
        parser.feed(fetch(row["url"]).decode("utf-8"))
        documents.append(
            dict(
                row, text=" ".join(parser.parts), capture_date=date.today().isoformat()
            )
        )
    save(output, documents)


def public_pack(source, output, path=None, count=1000, seed=2027):
    tasks, labels, documents = [], [], []
    if source == "confact":
        url = "https://raw.githubusercontent.com/zoeyyes/CONFACT/main/data/dataset/HumC.pkl.gz"
        content = Path(path).read_bytes() if path else fetch(url)
        rows = DataUnpickler(io.BytesIO(gzip.decompress(content))).load()
        for row in rows:
            identifier = "confact_" + str(row["id"])
            ids = []
            for evidence in row["evidence_url"]:
                doc_id = identifier + ":" + str(evidence["evidence_id"])
                ids.append(doc_id)
                documents.append(
                    {
                        "id": doc_id,
                        "text": evidence["content"],
                        "url": evidence["original_link"],
                        "capture_date": date.today().isoformat(),
                    }
                )
            tasks.append(
                {
                    "id": identifier,
                    "question": row["question"],
                    "dataset": "confact",
                    "split": "test",
                    "document_ids": ids,
                }
            )
            if row["label"] not in ("Supported", "Refuted"):
                raise ValueError(
                    "Unexpected native CONFACT label: " + str(row["label"])
                )
            labels.append(
                {
                    "id": identifier,
                    "answer": "Yes" if row["label"] == "Supported" else "No",
                }
            )
    else:
        from datasets import load_dataset

        rows = load_dataset("Warrieryes/CB_qa", split="train", streaming=True).shuffle(
            seed=seed, buffer_size=10000
        )
        kinds = ("misinformation", "temporal", "semantic")
        for number, row in enumerate(rows):
            if number >= count:
                break
            kind = kinds[number % 3]
            identifier = "conflictbank_" + str(number)
            evidence_key = (
                "misinformation_conflict_evidence_evidence"
                if kind == "misinformation"
                else kind + "_conflict_evidence"
            )
            ids = []
            for variant, text in (
                ("default", row["default_evidence"]),
                (kind, row[evidence_key]),
            ):
                doc_id = identifier + ":" + variant
                ids.append(doc_id)
                documents.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "capture_date": date.today().isoformat(),
                        "url": "https://huggingface.co/datasets/Warrieryes/CB_qa",
                    }
                )
            question = (
                row["question"]
                + "\n"
                + "\n".join(
                    f"{chr(65 + i)}. {text}" for i, text in enumerate(row["options"])
                )
            )
            tasks.append(
                {
                    "id": identifier,
                    "question": question,
                    "dataset": "conflictbank",
                    "split": "test",
                    "document_ids": ids,
                }
            )
            labels.append(
                {
                    "id": identifier,
                    "answer": row["correct_option"],
                    "category": kind,
                    "subject": row["subject"],
                    "relation": row["relation"],
                    "seed": seed,
                }
            )
    output = Path(output)
    save(output / "tasks.json", tasks)
    save(output / "labels.json", labels)
    save(output / "documents.json", documents)
