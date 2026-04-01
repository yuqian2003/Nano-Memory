import argparse
import json
import os
import sys
from tqdm import tqdm
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../")))
from evaluation.generation.metrics import evaluate_match, evaluate_sim
from async_llm import run_async
import asyncio
import aiohttp
import tiktoken
import time


PROMPT_G = """
You are an intelligent dialog bot. You will be shown History Dialogs. Please read, memorize, and understand the given Dialogs, then generate one concise, coherent and helpful response for the Question.

History Dialogs: {retrieved_texts}

Question Date: {question_date}
Question: {question}
"""


# locomo10 longmemeval_s longmemeval_m LongMTBench+
# python gen.py --dataset LongMTBench+ --retriever contriever --model_name_or_path gpt-4o-mini-2024-07-18 --topk 3 --method argmax

parser = argparse.ArgumentParser(description="long-term conversation evaluation")
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--retriever', type=str, required=True)
parser.add_argument("--model_name_or_path", type=str, default="gpt-4o-mini-2024-07-18")
parser.add_argument('--topk', type=int, required=True)
parser.add_argument('--method', type=str, required=True)

args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, f'logs/generation_logs/{args.dataset}-{args.retriever}-{args.method}-{args.model_name_or_path}-topk_{args.topk}.jsonl')
os.makedirs(os.path.dirname(save_path), exist_ok=True)


retrieved_data = []
retrieval_path = os.path.join(script_dir, f'logs/retrieval_logs/{args.dataset}-{args.retriever}-{args.method}.jsonl')
with open(retrieval_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        retrieved_data.append(json.loads(line.strip()))
data_path = os.path.join(script_dir, f'../data/process_data/{args.dataset}.json')
in_data = json.load(open(data_path, encoding="utf-8"))

################ session based
conv2sessions = {}
for entry in in_data:
    ids2session = {k:v for k,v in zip(entry['sessions_ids'],entry['sessions'])}
    conv2sessions[entry['conversation_id']] = ids2session
################


PROMPT_P = """
You are an intelligent dialog bot. You will be shown a "Fused Historical Event" which contains all the consolidated details relevant to your question.

Filter the information to extract only the parts directly relevant to the Question. Preserve original tokens, do not paraphrase. Remove irrelevant turns, redundant info, and non-essential details.

[Input Information]
Fused Historical Event: 
{fused_event}


Question Date: {question_date}
Question: {question}
Answer:
"""

generation_start = time.time()
encoding = tiktoken.get_encoding("cl100k_base")
prompt_token_counts = []
async_prompts = []

for idx, sample in enumerate(tqdm(retrieved_data)):
    conv_id = sample["conversation_id"]
    ids2session = conv2sessions[conv_id]

    # Build retrieved_texts with original sessions
    retrieved_texts = ""
    for retrieved_sess in sample['retrieval_results']['ranked_items'][:args.topk]:
        session = ids2session[retrieved_sess['corpus_id']]
        retrieved_texts += f"\n### Session Date: {retrieved_sess['timestamp']}\nSession Content:\n{session}\n"

    prompt = PROMPT_P.format(fused_event=retrieved_texts, question=sample["question"], question_date=sample["question_date"])
    async_prompts.append(prompt)
    prompt_token_counts.append(len(encoding.encode(prompt)))

async_responses = asyncio.run(run_async(async_prompts))
async_prompts = []

for idx, sample in enumerate(tqdm(retrieved_data)):
    # 第一轮 LLM 已经把 topk 个 session 过滤整合，直接使用其输出
    filtered_content = async_responses[idx]
    prompt = PROMPT_G.format(retrieved_texts=filtered_content, question=sample["question"], question_date=sample["question_date"])
    async_prompts.append(prompt)
    prompt_token_counts.append(len(encoding.encode(prompt)))

async_responses = asyncio.run(run_async(async_prompts))

results = []
for sample, response in zip(retrieved_data, async_responses):
    sample["response"] = response
    results.append(sample)
with open(save_path, "w", encoding="utf-8") as f:
    f.writelines([json.dumps(_, ensure_ascii=False) + "\n" for _ in results])

generation_time = time.time() - generation_start

def get_answer(ans):
    strip_word_list = [
        "\nDialogs:",
        "\n[bot]:",
        "\nAssistant:",
        "\nReview:",
        "\n",
        "[bot]:",
    ]
    cut_word_list = ["\n[human]:", "\nQuestion:", "\nQ:"]

    for strip_word in strip_word_list:
        if ans is not None:
            ans = ans.strip(strip_word)
        else:
            ans = ""
        
    for cut_word in cut_word_list:
        if cut_word in ans:
            ans = ans.split(cut_word)[0]
    return ans

print('Calculating metrics')
pred_all = []
for res in results:
    ans = get_answer(res["response"])
    pred_all.append(ans)

answer_all = []
for res in results:
    answer_all.append(str(res["answer"]))

metrics = evaluate_sim(pred_all, answer_all, truncate_pred=False)
metrics.update(evaluate_match(pred_all, answer_all, truncate_pred=False))

metrics["Avg.Token"] = round(sum(prompt_token_counts) / len(prompt_token_counts), 2)
metrics["Generation.Time"] = round(generation_time, 2)


print(metrics)
metrics_dir = os.path.join(os.path.dirname(save_path), "metrics")
os.makedirs(metrics_dir, exist_ok=True)
with open(
    os.path.join(
        metrics_dir, os.path.basename(save_path).replace("answer", "metrics")
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(metrics, f)
