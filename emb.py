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
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())  
            ids2granularity.update(data)
    return ids2granularity
    

class EmbeddingModelContriever():

    def __init__(self):
        self.model = AutoModel.from_pretrained('facebook/contriever').to(torch.device('cuda', 0))
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

    def get_emb_contriever(self, expansion_ids, expansion):
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        
        with torch.no_grad():
            all_docs_vectors = []
            dataloader = DataLoader(expansion, batch_size=64, shuffle=False)
            for batch in tqdm(dataloader):
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                cur_docs_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                all_docs_vectors.append(cur_docs_vectors)
            all_docs_vectors = torch.concat(all_docs_vectors, axis=0)
        if expansion_ids:
            ids2emb = {expansion_ids[i]: all_docs_vectors[i] for i in range(len(expansion_ids))}
            return ids2emb
        else:
            return all_docs_vectors


class EmbeddingModelSBERT():
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
        questions = [qa_item["question"] for qa_item in conversation["qa"]]
        sessions = ['\n'.join(session) for session in conversation["sessions"]]
        turns = [turn for session in conversation["sessions"] for turn in session]
        for sessid in conversation["sessions_ids"]:
            id = sessid if 'longmemeval' in dataset else f"convid-{str(conversation['conversation_id'])}-sessid-{sessid}"

        q_embs = emb_model.get_emb_contriever(None, questions)
        s_embs = emb_model.get_emb_contriever(None, sessions)
        t_embs = emb_model.get_emb_contriever(None, turns)
        
        embform = {
            "conversation_id": conversation['conversation_id'],
            "questions": q_embs,
            "sessions": s_embs,
            "turns": t_embs,
        }
        all_emb.append(embform)

    torch.save(all_emb, save_path)
    
    return all_emb


if __name__ == '__main__':
    emb_rawdata('longmemeval_m', 'contriever')
    # locomo10, longmemeval_s, LongMTBench+
