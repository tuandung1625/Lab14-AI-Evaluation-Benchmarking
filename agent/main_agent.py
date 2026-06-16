import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "does", "for",
    "from", "how", "if", "in", "is", "it", "list", "of", "or", "should",
    "the", "to", "too", "usage", "user", "users", "what", "when", "who",
    "with", "about", "handle", "policy", "rule", "rules", "exact",
}

SYNONYMS = {
    "2fa": "mfa",
    "multi": "mfa",
    "factor": "mfa",
    "two": "mfa",
    "bill": "billing",
    "bills": "invoice",
    "receipt": "invoice",
    "receipts": "invoice",
    "refunds": "refund",
    "recovery": "recoverable",
    "limit": "rate",
    "limits": "rate",
    "permission": "permissions",
    "permissions": "permission",
    "roles": "role",
    "notify": "notification",
    "notifications": "notification",
}


class MainAgent:
    """Small offline RAG agent used for the evaluation lab."""

    def __init__(self, version: str = "v2", top_k: int = 3):
        self.version = version
        self.top_k = top_k
        self.name = f"SupportAgent-{version}"
        self.docs = self._load_docs()

    def _load_docs(self) -> List[Dict]:
        path = Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _tokens(self, text: str) -> List[str]:
        tokens = []
        for token in TOKEN_RE.findall(text.lower()):
            if token in STOPWORDS:
                continue
            if len(token) > 4 and token.endswith("s"):
                token = token[:-1]
            tokens.append(token)
        if self.version == "v1":
            return tokens
        return [SYNONYMS.get(token, token) for token in tokens]

    def _score_doc(self, query_tokens: List[str], doc: Dict) -> float:
        doc_tokens = set(self._tokens(" ".join([doc["title"], doc["content"], " ".join(doc["keywords"])])))
        query_set = set(query_tokens)
        overlap = query_set.intersection(doc_tokens)
        score = float(len(overlap))

        # V2 rewards direct keyword matches and therefore ranks hard policy terms better.
        if self.version == "v2":
            keyword_tokens = set(self._tokens(" ".join(doc["keywords"])))
            score += 0.5 * len(query_set.intersection(keyword_tokens))
        return score

    def retrieve(self, question: str) -> List[Tuple[Dict, float]]:
        query_tokens = self._tokens(question)
        scored = [(doc, self._score_doc(query_tokens, doc)) for doc in self.docs]
        scored.sort(key=lambda item: (-item[1], item[0]["id"]))
        return [item for item in scored if item[1] > 0][: self.top_k]

    def _compose_answer(self, question: str, retrieved: List[Tuple[Dict, float]]) -> str:
        if not retrieved:
            return "I do not know based on the available documentation."

        question_lower = question.lower()
        if "ignore" in question_lower or "invent" in question_lower or "poem" in question_lower:
            prefix = "I cannot ignore the documentation or invent policy details. "
        else:
            prefix = ""

        evidence = " ".join(doc["content"] for doc, _ in retrieved[:2])
        return prefix + evidence

    async def query(self, question: str) -> Dict:
        # Keeps the async runner realistic without making local tests slow.
        await asyncio.sleep(0.02 if self.version == "v2" else 0.04)
        retrieved = self.retrieve(question)
        answer = self._compose_answer(question, retrieved)
        token_estimate = max(30, len(question.split()) + len(answer.split()))

        return {
            "answer": answer,
            "contexts": [doc["content"] for doc, _ in retrieved],
            "retrieved_ids": [doc["id"] for doc, _ in retrieved],
            "metadata": {
                "agent_version": self.version,
                "model": "offline-keyword-rag",
                "tokens_used": token_estimate,
                "estimated_cost_usd": round(token_estimate * 0.0000005, 6),
                "sources": [doc["title"] for doc, _ in retrieved],
            },
        }


if __name__ == "__main__":
    async def test() -> None:
        agent = MainAgent()
        print(await agent.query("How do I reset my password?"))

    asyncio.run(test())
