"""
Legal Needle Bench RAG evaluation code.

- Loads FAISS DB per embedding model
- Loads QA split
- For each (emb × gen × k × question): retrieve → answer → judge → write JSONL
"""

import json
import os
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from config import DB_DIR, embedding_models, generative_models, judge_model, Ks
from prompts import RAG_PROMPT, JUDGE_PROMPT


# --------------------
# IO and loading
# --------------------
def load_db(emb_cfg: Dict[str, Any]) -> FAISS:
    """Load the persisted FAISS DB for an embedding model config."""
    return FAISS.load_local(
        os.path.join(DB_DIR, emb_cfg["model_name"]),
        embeddings=emb_cfg["model"],
        allow_dangerous_deserialization=True,
    )


def load_qa(split: str):
    """Load the QA evaluation split."""
    return load_dataset("isaacus/legal-rag-bench", "qa", token=os.environ["HF_TOKEN"], split=split)


def ensure_fresh_file(path: str) -> None:
    """Create parent dirs and reset the output file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a JSON-serializable dict to a .jsonl file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# --------------------
# RAG and LLM as a judge
# --------------------
def format_context(docs) -> str:
    """Format retrieved docs into the context block expected by the prompt."""
    return "\n\n".join(
        f"DOC {i}\nMETADATA: {d.metadata}\nCONTENT: {d.page_content}"
        for i, d in enumerate(docs)
    )


def docs_to_json(docs) -> List[Dict[str, Any]]:
    """Make docs JSON-friendly for logging."""
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]


def relevant_passage_in_context(rel_id: str, docs) -> bool:
    """Checks whether the relevant passage id appears in retrieved doc metadata."""
    return any(rel_id in d.metadata.get("id", "") for d in docs)


def rag_answer(db: FAISS, llm, question: str, k: int) -> Tuple[str, Any, str]:
    """Retrieve, build prompt, and generate answer."""
    docs = db.similarity_search(question, k=k)
    context = format_context(docs)
    answer = llm.invoke(RAG_PROMPT.format(question=question, context=context)).content
    return answer, docs, context


def judge_answer(judge_llm, question: str, rag_answer: str, context: str, answer: str) -> Dict[str, Any]:
    """Judge answer; return parsed JSON or a fallback with raw output."""
    verdict_text = judge_llm.invoke(
        JUDGE_PROMPT.format(
            question=question,
            rag_answer=rag_answer,
            answer=answer,
            context=context,
        )
    ).content
    try:
        return json.loads(verdict_text)
    except Exception:
        return {"grounded": None, "correct": None, "reasoning": None, "raw": verdict_text}


# --------------------
# Record building
# --------------------
def build_record(
    *,
    k: int,
    row: Dict[str, Any],
    emb_name: str,
    gen_name: str,
    rag_ans: str,
    verdict: Dict[str, Any],
    docs,
) -> Dict[str, Any]:
    """Create a single result record."""
    qid = row["id"]
    model_name = f"{emb_name}_{gen_name}"
    result_id = f"{k}-{qid}"

    return {
        "id": f"{model_name}_{result_id}",
        "model_name": model_name,
        "embedding_model": emb_name,
        "generative_model": gen_name,
        "result_id": result_id,
        "question_id": qid,
        "question": row["question"],
        "answer": row["answer"],
        "rag_answer": rag_ans,
        "judge_verdict": verdict,
        "context": docs_to_json(docs),
        "relevant_passage_id": row["relevant_passage_id"],
        "relevant_passage_in_context": relevant_passage_in_context(row["relevant_passage_id"], docs),
    }


# --------------------
# Runs full iteration for each (emb × gen × k × question): retrieve → answer → judge → write JSONL
# --------------------
def run_all(qa_split: str = "test", limit: int | None = None) -> None:
    qa = load_qa(qa_split)

    # Set limit = n<100 to run tests
    if limit is not None:
        qa = qa.select(range(limit))

    judge_llm = judge_model["model"]

    results_root = "results"
    os.makedirs(results_root, exist_ok=True)

    for emb_cfg in tqdm(embedding_models, desc="Embedding models"):
        emb_name = emb_cfg["model_name"]
        db = load_db(emb_cfg)

        for gen_cfg in tqdm(generative_models, desc=f"{emb_name}: Generative models", leave=False):
            gen_name = gen_cfg["model_name"]
            gen_llm = gen_cfg["model"]

            out_path = os.path.join(results_root, emb_name, f"{emb_name}_{gen_name}_results.jsonl")
            ensure_fresh_file(out_path)

            for k in tqdm(Ks, desc=f"{emb_name}/{gen_name}: Ks", leave=False):
                for row in tqdm(qa, desc=f"{emb_name}/{gen_name}: Qs (k={k})", leave=False):
                    rag_ans, docs, ctx = rag_answer(db, gen_llm, row["question"], k)
                    verdict = judge_answer(judge_llm, row["question"], rag_ans, ctx, row["answer"])

                    rec = build_record(
                        k=k,
                        row=row,
                        emb_name=emb_name,
                        gen_name=gen_name,
                        rag_ans=rag_ans,
                        verdict=verdict,
                        docs=docs,
                    )
                    append_jsonl(out_path, rec)

            tqdm.write(f"Wrote: {out_path}")


if __name__ == "__main__":
    run_all(qa_split="test")
