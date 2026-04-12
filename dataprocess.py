import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_longmemeval_s():

    in_data = json.load(open('data/unprocess/longmemeval_s.json'))
    alldata = []

    for entry in tqdm(in_data):
        question_id = entry['question_id']
        question_type = entry['question_type']
        question = entry['question']
        answer = entry['answer']
        question_date = entry['question_date']
        haystack_dates = entry['haystack_dates']
        haystack_session_ids = entry['haystack_session_ids']
        haystack_sessions = entry['haystack_sessions']
        answer_session_ids = []
        for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
            for turn_id, turn in enumerate(sess_entry):
                if 'has_answer' in turn and turn['has_answer']==True:
                    answer_session_ids.append(f"{cur_sess_id.replace('answer_','')}")

        sessions = []
        for sess_entry in entry['haystack_sessions']:
            session = []
            for item in sess_entry:
                session.append(f"[{item['role']}]: {item['content']}")
            merged_session = []
            for i in range(0, len(session), 2):
                if i + 1 < len(session):
                    merged_session.append(session[i] + "\n" + session[i+1])
                else:
                    merged_session.append(session[i])
            if len(merged_session)==0:
                print(len(merged_session),len(session))
            sessions.append(merged_session)
            
        dataform = {
            'conversation_id':entry['question_id'],
            'qa':[
                {
                "question": entry['question'],
                "question_type": entry['question_type'],
                "question_date":entry['question_date'],
                "answer": entry['answer'],
                "answer_session_ids": answer_session_ids,
            },
            ],
            'sessions_ids':[s.replace('answer_', '') for s in entry['haystack_session_ids']],
            'sessions_dates':entry['haystack_dates'],
            'sessions':sessions
            }
        alldata.append(dataform)
        
    with open("./process_data/longmemeval_s.json", "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)


def process_longmemeval_m():

    in_data = json.load(open('data/unprocess/longmemeval_m.json'))
    alldata = []

    for entry in tqdm(in_data):
        question_id = entry['question_id']
        question_type = entry['question_type']
        question = entry['question']
        answer = entry['answer']
        question_date = entry['question_date']
        haystack_dates = entry['haystack_dates']
        haystack_session_ids = entry['haystack_session_ids']
        haystack_sessions = entry['haystack_sessions']
        answer_session_ids = []
        for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
            for turn_id, turn in enumerate(sess_entry):
                if 'has_answer' in turn and turn['has_answer']==True:
                    answer_session_ids.append(f"{cur_sess_id.replace('answer_','')}")

        sessions = []
        for sess_entry in entry['haystack_sessions']:
            session = []
            for item in sess_entry:
                session.append(f"[{item['role']}]: {item['content']}")
            merged_session = []
            for i in range(0, len(session), 2):
                if i + 1 < len(session):
                    merged_session.append(session[i] + "\n" + session[i+1])
                else:
                    merged_session.append(session[i])
            if len(merged_session)==0:
                print(len(merged_session),len(session))
            sessions.append(merged_session)
            
        dataform = {
            'conversation_id':entry['question_id'],
            'qa':[
                {
                "question": entry['question'],
                "question_type": entry['question_type'],
                "question_date":entry['question_date'],
                "answer": entry['answer'],
                "answer_session_ids": answer_session_ids,
            },
            ],
            'sessions_ids':[s.replace('answer_', '') for s in entry['haystack_session_ids']],
            'sessions_dates':entry['haystack_dates'],
            'sessions':sessions
            }
        alldata.append(dataform)
        
    with open("./process_data/longmemeval_m.json", "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)

def process_locomo10():
    in_data = json.load(open('data/unprocess/locomo10.json'))

    alldata = []

    for entry in tqdm(in_data):
        newqa = []
        for qaitem in entry['qa']:
            if 'adversarial_answer' in qaitem:
                answer = qaitem['adversarial_answer']
            else:
                answer = qaitem['answer']
            answer_session_ids = []
            for item in qaitem['evidence']:
                try:
                    turn_id = int(item.split(':')[1])
                except:
                    continue
                answer_session_ids.append(f"{item.replace('D', 'session_').split(':')[0]}")

            newqa.append(
                {
                "question": qaitem['question'],
                "question_type": qaitem['category'],
                "question_date": None,
                "answer": answer,
                "answer_session_ids":answer_session_ids,
            })
            
        conversation = entry['conversation']
        sessions_ids = []
        sessions_dates = []
        sessions = []
        for i in range(1000):
            if f'session_{i+1}' in conversation:
                sessions_ids.append(f'session_{i+1}')
                sessions_dates.append(conversation[f'session_{i+1}_date_time'])
                session = []
                for dialog in conversation[f'session_{i+1}']:
                    if 'blip_caption' in dialog:
                        session.append(f"[{dialog['speaker']}]: {dialog['text']}\n The image Caption: {dialog['blip_caption']}")
                    else:
                        session.append(f"[{dialog['speaker']}]: {dialog['text']}")
                merged_session = []
                for i in range(0, len(session), 2):
                    if i + 1 < len(session):
                        merged_session.append(session[i] + "\n" + session[i+1])
                    else:
                        merged_session.append(session[i])

                sessions.append(merged_session)
        dataform = {
            'conversation_id':entry['sample_id'],
            'qa':newqa,
            'sessions_ids':sessions_ids,
            'sessions_dates':sessions_dates,
            'sessions':sessions
            }
        alldata.append(dataform)
        

    with open("./process_data/locomo10.json", "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)
        
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool8)):
        return bool(obj)
    # Check for pandas/None values
    elif pd.isna(obj):
        return None
    # Recursively handle dicts and lists
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def process_LongMTBench_plus():
    parquet_path = 'data/unprocess/test-00000-of-00001.parquet'
    df = pd.read_parquet(parquet_path)
    
    # Convert DataFrame to list of dictionaries and handle numpy types
    raw_data = df.to_dict('records')
    raw_data = [convert_numpy_types(record) for record in raw_data]
    
    # Process data into final format
    alldata = []
    for entry in tqdm(raw_data):
        # dict_keys(['sessions', 'questions', 'conversation_id', 'turns', 'answers'])
        newqa = []
        for q, a in zip(entry['questions'], entry['answers']):
            newqa.append({
                "question": q,
                "question_type": None,
                "question_date": None,
                "answer": a,
                "answer_session_ids": None,
            })
        sessions_ids = []
        sessions_dates = []
        for i, session in enumerate(entry['sessions']):
            sessions_ids.append(f'session_{i+1}')
            sessions_dates.append(None)
        dataform = {
            'conversation_id': entry['conversation_id'],
            'qa': newqa,
            'sessions_ids': sessions_ids,
            'sessions_dates': sessions_dates,
            'sessions': entry['sessions']
        }
        alldata.append(dataform)
    
    # Save processed data to process_data directory
    with open("./process_data/LongMTBench+.json", "w", encoding="utf-8") as f:
        json.dump(alldata, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    os.makedirs("./process_data/", exist_ok=True)
    process_locomo10()
    process_LongMTBench_plus()
    process_longmemeval_s()
    process_longmemeval_m()
