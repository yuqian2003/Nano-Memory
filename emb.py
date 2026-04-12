import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer


def read_ids2granularity(path):
    ids2granularity = {}
    #
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())  
            ids2granularity.update(data)
    return ids2granularity
    

class EmbeddingModelContriever():

    def __init__(self):
        # 初始化 Facebook 开源的 Contriever 模型并部署到 GPU (cuda:0)
        self.model = AutoModel.from_pretrained('facebook/contriever').to(torch.device('cuda', 0))
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

    def get_emb_contriever(self, expansion_ids, expansion):
        # 定义 Mean Pooling：将 Token 级别的向量加权平均，得到代表整句的向量
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        
        with torch.no_grad(): # 禁用梯度计算以节省显存
            all_docs_vectors = []
            # 使用 DataLoader 批量处理文本，batch_size=64 提高速度
            dataloader = DataLoader(expansion, batch_size=64, shuffle=False)
            for batch in tqdm(dataloader):
                # 文本分词并移动到 GPU
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                # 模型推理
                outputs = self.model(**inputs)
                # 执行 Pooling 得到句子向量并转回 CPU
                cur_docs_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                all_docs_vectors.append(cur_docs_vectors)
            all_docs_vectors = torch.concat(all_docs_vectors, axis=0)
        
        # 如果提供了 ID，返回字典映射；否则直接返回向量张量
        if expansion_ids:
            ids2emb = {expansion_ids[i]: all_docs_vectors[i] for i in range(len(expansion_ids))}
            return ids2emb
        else:
            return all_docs_vectors


class EmbeddingModelSBERT():
    '''
    向量提取模型类: SBERT(常用于句向量)
    '''
    def __init__(self, retriever):
        if retriever == 'mpnet':
            self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
        elif retriever == 'minilm':
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
    def get_emb_contriever(self, expansion_ids, expansion):
        all_docs_vectors = self.model.encode(expansion)
        if expansion_ids:
            ids2emb = {expansion_ids[i]: all_docs_vectors[i] for i in range(len(expansion_ids))}
            return ids2emb
        else:
            return torch.tensor(all_docs_vectors)


def emb_rawdata(dataset, retriever):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    emb_dir = os.path.join(script_dir, 'logs/process_embs')
    os.makedirs(emb_dir, exist_ok=True)
    save_path = os.path.join(emb_dir, f'{dataset}-{retriever}-emb.pt')
    data_path = f'../data/process_data/{dataset}.json'
    in_data = json.load(open(data_path))

    if retriever == 'contriever':
        emb_model = EmbeddingModelContriever()
    else:
        emb_model = EmbeddingModelSBERT(retriever)

    all_emb = []
    for conversation in tqdm(in_data):
        # (1) Questions: 问题
        questions = [qa_item["question"] for qa_item in conversation["qa"]]
        # (2) Sessions: 整个会话文本
        sessions = ['\n'.join(session) for session in conversation["sessions"]]
        # (3) Turns: 每一轮对话的具体内容
        turns = [turn for session in conversation["sessions"] for turn in session]


        for sessid in conversation["sessions_ids"]:
            # 构造 ID 来从字典中查找对应的摘要和关键词
            id = sessid if 'longmemeval' in dataset else f"convid-{str(conversation['conversation_id'])}-sessid-{sessid}"

        q_embs = emb_model.get_emb_contriever(None, questions)
        s_embs = emb_model.get_emb_contriever(None, sessions)
        t_embs = emb_model.get_emb_contriever(None, turns)
        
        # 6. 构造存储结构
        embform = {
            "conversation_id": conversation['conversation_id'],
            "questions": q_embs,
            "sessions": s_embs,
            "turns": t_embs,
        }
        all_emb.append(embform)
        
    # 7. 将所有对话的向量数据保存为 PyTorch 格式
    torch.save(all_emb, save_path)
    
    return all_emb


if __name__ == '__main__':
    #emb_rawdata('LongMTBench+', 'contriever')
    #emb_rawdata('LongMTBench+', 'mpnet')
    #emb_rawdata('LongMTBench+', 'minilm')
    #emb_rawdata('longmemeval_s', 'contriever')
    #emb_rawdata('longmemeval_s', 'mpnet')
    #emb_rawdata('longmemeval_s', 'minilm')
    emb_rawdata('longmemeval_m', 'contriever')
