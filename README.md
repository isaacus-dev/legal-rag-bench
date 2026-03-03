# Legal RAG Bench

This repository contains the code used to evaluate RAG systems on Legal RAG Bench.

If you're looking for the data behind Legal RAG Bench, you can find it [here](https://huggingface.co/datasets/isaacus/legal-rag-bench). A full interactive write up of how Legal RAG Bench was built is also available [here](https://isaacus.com/blog/legal-rag-bench).

## Setup

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

We recommend creating a `.env` file in the repository root to store API keys for the providers you plan to use:

```env
# Isaacus
ISAACUS_API_KEY=...

# OpenAI
OPENAI_API_KEY=...

# Google
GOOGLE_API_KEY=...
```

Replace `...` with your actual keys. You can omit any providers you won’t be using.

## Basic Usage

The default configuration evaluates **n = 6** RAG pipeline permutations:

1. **Embedding models** (retrieval at `k=5` only):
   - `kanon-2-embedder`
   - `text-embedding-3-large`
   - `gemini-embedding-001`
2. **Generative models**:
   - `gpt-5.2`
   - `gemini-3.1-pro-preview`
3. **Judge LLM**:
   - `gpt-5.2` with `reasoning_effort="high"`

Counts:
- **Runs** = emb × gen × nK (default: 3 × 2 × 1 = 6)
- **Iterations** = emb × gen × nK × nQuestions (default: 3 × 2 × 1 × 100 = 600)

To run the default evaluation end-to-end:

1. **Build vector DBs** (one per embedding model):
   ```bash
   python db.py
   ```

2. **Run the evaluation** over the full QA dataset for each emb × gen pairing:
   ```bash
   python eval.py
   ```

3. **Inspect results** in `./results`:
   - Results are organized into folders per embedding model
   - Each embedding × generative pairing produces a `.jsonl` file

Each `.jsonl` contains one row per question (and per `k`, if multiple `k` values are used). Rows include metadata (labels, IDs), the RAG answer, the human-annotated answer, retrieved context documents, and the judge’s assessment.

To extract the judge outcome, read the `"judge_verdict"` field:
- `correct`: `true`/`false` — whether the judge deemed the answer correct given the human-annotated answer
- `grounded`: `true`/`false` — whether the judge deemed the answer grounded in the provided context
- `reasoning`: `str` — the judge’s explanation for correctness and grounding

The `"relevant_passage_in_context"` field is also useful for determining if the retrieval model was able to deliver the relevant passage as context to the generative model.

## Advanced Usage

Before running `db.py` and `eval.py`, you can tune or evaluate a custom configuration via the following knobs.

### Select different embedding models, generative models, judges, or `k` values

1. **Edit `config.py`** to define new embedding and/or generative models using LangChain integrations. For example:

   ```python
   # Example embedding model
   kanon2 = {
       "model": IsaacusEmbeddings(
           model="kanon-2-embedder",
           api_key=ISAACUS_API_KEY,
       ),
       "model_name": "kanon2",
   }

   # Example generative model
   gpt52 = {
       "model": ChatOpenAI(
           model="gpt-5.2",
           api_key=OPENAI_API_KEY,
           temperature=0,
           reasoning_effort="none",
           seed=42,
       ),
       "model_name": "gpt52",
   }
   ```

   If your changes require additional LangChain packages or new provider API keys, update imports and your `.env` accordingly.

2. **Add your models** to the lists the evaluation iterates over:

   ```python
   # For embedding models, add new models to this list
   embedding_models = [kanon2, ...]

   # For generative models, add new models to this list
   generative_models = [gpt52, ...]
   ```

3. **Change the judge model** by assigning a different LangChain LLM to `judge_model`:

   ```python
   judge_model = {
       "model": ChatOpenAI(
           model="gpt-5.2",
           api_key=OPENAI_API_KEY,
           reasoning_effort="high",
           seed=42,
       ),
       "model_name": "gpt52_judge",
   }
   ```

4. **Evaluate different retrieval depths** by editing the `Ks` list:

   ```python
   # Number of retrieved documents provided as context to the generative model
   Ks = [5, ...]
   ```

### Use different prompts for the generator or judge

Edit `prompts.py`:
- Update `RAG_PROMPT` to change the generative model prompt/style
- Update `JUDGE_PROMPT` to change the judge rubric/prompt

The default prompts are tuned for strong performance on the Legal RAG Bench corpus.

## Reproducibility
The results reported for Legal RAG Bench were generated on the 20th of Feburary 2026, shortly after the public preview release of Gemini 3.1 Pro. You may observe different outputs and/or benchmark scores across runs, even with the same prompts and data, for several reasons:
1. **LLM outputs are non-deterministic.** APIs can return different outputs across requests with identical inputs. Setting `temperature=0` may reduce variance but does not guarantee determinism. If your provider supports it, consider using a fixed `seed` and recording any response metadata to aid reproducibility. Note that doing this will deviate from the default settings tested in Legal RAG Bench.
2. **Model providers update models over time.** A model “name” or alias may refer to an evolving system. Provider-side updates (weights, routing, safety layers, tool policies, decoding defaults, etc.) can change behavior and benchmark scores between dates, sometimes without a clear semantic-version signal.
3. **LangChain and other dependencies change quickly.** Updates can modify prompt formatting, tokenization, retriever defaults, retry logic, and other behaviors that materially affect results.
4. **Runtime/infrastructure differences matter.** Python versions, OS, vector DB implementations, embedding libraries, and hardware can affect retrieval behavior, latency, and occasionally outputs.

## License

This project is licensed under the [MIT](LICENSE) license. 

## Citation

```bibtex
@misc{butler2026legalragbench,
      title={Legal RAG Bench: an end-to-end benchmark for legal RAG}, 
      author={Abdur-Rahman Butler and Umar Butler},
      year={2026},
      eprint={2603.01710},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.01710}, 
}
```
