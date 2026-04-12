import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, src_path)
sys.path.insert(0, os.path.join(src_path, "evaluation/retrieval"))
import multiprocessing as mp
from functools import partial
import json
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from eval_utils import evaluate_retrieval
from emb import emb_rawdata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='locomo10') # , required=True
    parser.add_argument('--retriever', type=str, default='contriever') # , required=True
    parser.add_argument('--method', type=str, default='argmax') # , required=True
    parser.add_argument('--topk_turns', type=int, default=1,
                        help='Number of top turns to sum per session (1=argmax/max, >1=sum of top-K)')

    return parser.parse_args()

# contriever / minilm / mpnet

def main(args):
    build_start = time.time()

    emb_dir = os.path.join(current_dir, "logs", "process_embs")
    emb_path = os.path.join(emb_dir, f"{args.dataset}-{args.retriever}-emb.pt")
    if os.path.exists(emb_path):
        all_emb = torch.load(emb_path)
    else:
        all_emb = emb_rawdata(args.dataset, args.retriever)

    in_data = json.load(open(f'../data/process_data/{args.dataset}.json'))

    conv_turn_embeddings = {}
    for entry, emb in zip(in_data, all_emb):
        assert entry['conversation_id'] == emb['conversation_id']
        conv_id = entry['conversation_id']

        turn_num_each_session = [len(sess) for sess in entry['sessions']]
        turn_embeddings = []
        start_idx = 0
        for num_turns in turn_num_each_session:
            if num_turns == 0:
                turn_mean_emb = torch.zeros(emb['turns'].size(1))
            else:
                session_turn_embs = emb['turns'][start_idx:start_idx + num_turns]
                turn_mean_emb = session_turn_embs.mean(dim=0)
            turn_embeddings.append(turn_mean_emb)
            start_idx += num_turns
        conv_turn_embeddings[conv_id] = torch.stack(turn_embeddings)

    build_time = time.time() - build_start

    retrieval_start = time.time()

    results = []
    for entry, emb in zip(in_data, all_emb):
        conv_id = entry['conversation_id']
        turn_embeddings = conv_turn_embeddings[conv_id]
        turn_num_each_session = [len(sess) for sess in entry['sessions']]

        for qa_one, q_emb in zip(entry['qa'], emb['questions']):
            if args.dataset != "LongMTBench+":
                correct_docs = list(set([ids for ids in qa_one['answer_session_ids']]))

            if args.method == 'session_level':
                scores = (q_emb @ emb['sessions'].T).squeeze()
                rankings = scores.argsort(descending=True)
            
            elif args.method == 'turn_level':
                # turn agg 把每个seesion内所有tune合并平均为一个向量，得到所有session向量，再做余弦相似度,排序session分数
                scores = (q_emb @ turn_embeddings.T).squeeze()
                rankings = scores.argsort(descending=True)
            
            elif args.method == 'argmax':
                # Each session's score = sum of its top-K turn scores (K=1 reduces to argmax/max)
                scores = (q_emb @ emb['turns'].T).squeeze()

                n_sessions = len(turn_num_each_session)
                session_scores = torch.zeros(n_sessions)
                start_idx = 0
                for sess_idx, num_turns in enumerate(turn_num_each_session):
                    if num_turns == 0:
                        session_scores[sess_idx] = 0.0
                    else:
                        sess_turn_scores = scores[start_idx:start_idx + num_turns]
                        k_actual = min(args.topk_turns, num_turns)
                        topk_vals, _ = torch.topk(sess_turn_scores, k=k_actual)
                        session_scores[sess_idx] = topk_vals.sum()
                    start_idx += num_turns

                rankings = session_scores.argsort(descending=True)
            
            cur_results = {
                "conversation_id": entry['conversation_id'],
                'question_type': qa_one['question_type'],
                'question': qa_one['question'],
                'answer': qa_one['answer'],
                'question_date': qa_one['question_date'],
                'retrieval_results': {
                    'ranked_items': [
                        {
                            # 'corpus_id': turn_ids[rid], 
                            'corpus_id': entry['sessions_ids'][rid],
                            'timestamp': entry['sessions_dates'][rid],
                        }
                        for rid in rankings
                    ],
                    'metrics': {
                        'session': {},
                        'turn': {}
                    }
                }
            }

            # 7. 计算评价指标：Recall 和 NDCG
            if args.dataset != "LongMTBench+":
                # for k in [1, 3, 5, 10, 30, 50]:
                for k in [3, 5, 10]:
                    recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, entry['sessions_ids'], k=k)
                    cur_results['retrieval_results']['metrics']['session'].update({
                        'recall_any@{}'.format(k): recall_any,
                        'recall_all@{}'.format(k): recall_all,
                        'ndcg_any@{}'.format(k): ndcg_any
                    })
            results.append(cur_results)

    retrieval_time = time.time() - retrieval_start

    if args.dataset != "LongMTBench+":
        refine_results = []
        for k in results[0]['retrieval_results']['metrics']['session']:
            k_result = np.mean([x['retrieval_results']['metrics']['session'][k] for x in results if '_abs' not in str(x['conversation_id'])])
            if k.startswith("recall_all@") or k.startswith("ndcg_any@"):
                refine_results.append(f"{round(k_result*100, 2)}")
        print("\t".join(refine_results))
    
    out_dir = os.path.join(current_dir, "logs", "retrieval_logs")
    os.makedirs(out_dir, exist_ok=True)
    method_tag = args.method if args.topk_turns == 1 else f"{args.method}_top{args.topk_turns}"
    out_file = os.path.join(out_dir, f"{args.dataset}-{args.retriever}-{method_tag}.jsonl")
    out_f = open(out_file, "w")
    for entry in results:
        print(json.dumps(entry), file=out_f)
    out_f.close()

    print(f'\n===== Build time: {build_time:.2f}s =====')
    print(f'===== Retrieval time: {retrieval_time:.2f}s =====')


if __name__ == '__main__':
    args = parse_args()
    main(args)
