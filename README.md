# Nano-Memory

Implementation for paper: Back to Basics: Let Conversational Agents Remember with Just Retrieval and Generation

## Dataset

We evaluate our method on four publicly available long-context conversation benchmarks specifically designed to test agents' ability to handle long-term conversational dependencies:

- **LoCoMo**
- **Long-MT-Bench+**
- **LongMemEval-s**
- **LongMemEval-m**

### Download & Preprocessing

1. Download the original datasets from the following links:

   - [LongMemEval-s & LongMemEval-m](https://github.com/xiaowu0162/LongMemEval)
   - [LoCoMo-10](https://github.com/snap-research/locomo/blob/main/data/locomo10.json)
   - [Long-MT-Bench+](https://huggingface.co/datasets/panzs19/Long-MT-Bench-Plus)

2. Place all the downloaded files into the `data/unprocess/` directory.

3. Run the preprocessing script:

```bash
   cd data
   python dataprocess.py
 ```

## Environment
    conda create -n nano-memory python=3.9
    conda activate nano-memory
    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt

## Experiments

```bash
python3 emb.py
 ```

### Memory Retrieval

```bash
python3 run_retrieval.py --dataset locomo10 --retriever contriever --method argmax
 ```

Change the --dataset parameter to  `locomo10` `longmemeval_s` `longmemeval_m` `LongMTBench+` for experiments on other datasets.
Change the --retriever parameter to  `contriever` `mpnet` `minilm` for experiments on other retrievers.


### Memory Generation

To conduct QA experiements in Tab.1, run:

```bash
python generation.py --dataset locomo10 --retriever contriever --model_name_or_path gpt-4o-mini-2024-07-18 --topk 3 --method memgas
 ```


#### Evaluation

Evaluating with GPT4o-as-Judge:

```bash
python 4o_J.py --model_name_or_path gpt-4o --eval_file locomo10-contriever-argmax-gpt-4o-mini-2024-07-18-topk_3.jsonl

```

Evaluating different query types:
```python
python eval_query_type.py --eval_file logs/generation_logs/locomo10-contriever-argmax-gpt-4o-mini-2024-07-18-topk_3.jsonl
```
