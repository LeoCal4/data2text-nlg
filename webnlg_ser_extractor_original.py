from typing import List, Union, Tuple
import re
import os
import json


def clean(entity):
    entity = entity.lower().replace('_', ' ')
    # separate punct signs from text
    entity = ' '.join(re.split('(\W)', entity))
    entity = ' '.join(entity.split())  # delete whitespaces
    return entity


def clean_mr(mr):
    # (19255)_1994_VK8 | density | 2.0(gramPerCubicCentimetres) | | |
    # extract all subjects and objects and clean them
    subj_obj = []
    triples = mr.strip().split('|||')  # the last one is empty
    triples = [triple for triple in triples if triple]  # delete empty triples
    for triple in triples:
        s, _, o = triple.split(' | ')
        s = subj_obj.append(clean(s))
        o = subj_obj.append(clean(o))
    return subj_obj


def extract_and_clean_subjects_and_objects_from_mr(mr: str, datatuner_format: bool = False) -> Tuple[List, List]:
    #* extract all subjects and objects and clean them
    subjects = []
    objects = []
    mr = mr.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    if datatuner_format:
        separated_mrs = mr.split(" ; ")
    else:
        separated_mrs = mr.split("|||")
    for sep_mr in separated_mrs:
        if sep_mr == "" or sep_mr == " ":
            continue
        try:
            if datatuner_format:
                subject, _, obj = re.findall(r"<[\w\s]*>\s*([^<;]*)(?:;)?", sep_mr.strip())
            else:
                subject, _, obj = sep_mr.strip().split("|")
        except:
            print("ERROR: ", sep_mr)
        #* delete white spaces
        subjects.append(clean(subject))
        objects.append(clean(obj))
    return subjects, objects


def get_all_subj_obj(base_dataset_path: str):
    #* read all the webnlg corpus
    complete_dataset = []
    for partition in ["train", "validation", "test"]:
        partition_path = os.path.join(base_dataset_path, f"{partition}.json")
        with open(partition_path, "r", encoding="utf-8") as f:
            complete_dataset += json.load(f)
    complete_mrs = [entry["raw_modifiedtripleset"] for entry in complete_dataset]
    subjects = []
    objects = []
    #* extract and clean all subjects and objects
    for mr in complete_mrs: # mr = meaning representation aka the data
        subject, obj = extract_and_clean_subjects_and_objects_from_mr(mr)
        subjects.extend(subject)
        objects.extend(obj)
    return subjects, objects


def calculate_ser(mrs_raw: List[str], utterances: List[str], base_dataset_path: str):
    #* get all cleaned s and o from the whole corpus
    all_subj_cleaned, all_obj_cleaned = get_all_subj_obj(base_dataset_path)
    entities = list(set(all_subj_cleaned + all_obj_cleaned))
    #* delete all numbers from entities
    for i, entity in enumerate(entities):
        try:
            float(entity)
            del entities[i]
        except ValueError:
            pass
    se_rates = []
    wrong_entries = []
    for mr, pred in zip(mrs_raw, utterances):
        missing_values = []
        hallucinated_values = []
        values = clean_mr(mr)
        total_n_slots = len(values)
        missing = 0
        hallucinated = 0
        for value in values:
            if value not in pred.lower():
                missing += 1
                missing_values.append(value)
        # delete s and o that are present in MR
        all_subj_obj_not_pres = [item for item in entities if item not in values]
        for entity in all_subj_obj_not_pres:
            if entity in pred.split():
                hallucinated += 1
                hallucinated_values.append(entity)
        ser = (missing + hallucinated) / total_n_slots
        se_rates.append(ser)
        if missing > 0 or hallucinated > 0:
            wrong_entries.append({
                "mr": mr, 
                "pred": pred, 
                "n_missing": missing, 
                "missing": missing_values,
                "n_hallucinated": hallucinated, 
                "hallucinated_values": hallucinated_values})
    with open("webnlg_og_ser_results.json", "w", encoding="utf-8") as f:
        json.dump(wrong_entries, f, ensure_ascii=False, indent=4, sort_keys=False)
    ser = sum(se_rates) / len(se_rates)
    uer = sum([curr_ser > 0 for curr_ser in se_rates]) / len(se_rates)
    return ser, -1, uer, -1
    # av_ser_in_percentage = (sum(se_rates) / len(se_rates)) * 100
    # print(round(av_ser_in_percentage, 2), '%')


if __name__ == "__main__":
    base_dataset_path = r"C:\Users\Leo\Documents\PythonProjects\Tesi\datatuner\data\webnlg"
    with open(os.path.join(base_dataset_path, "train.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)
    mrs = [entry["raw_modifiedtripleset"] for entry in dataset]
    
    #* DATATUNER
    # print("DT")
    # utt_path = r"C:\Users\Leo\Desktop\Tesi\results\webnlg\webnlg_base_results\reranked.json"
    # utterances = extract_original_datatuner_refs(utt_path)
    
    #* T5 (VERY OLD)
    # print("T5 (OLD)")
    # utt_path = r"C:\Users\Leo\Desktop\Tesi\custom predictions\webnlg_default_choice\webnlg_t5_generated.txt"
    # with open(utt_path, "r", encoding="utf-8") as f:
    #     raw_utts = f.readlines()
    # utterances = [utt.split("GEN (default): ")[1] for utt in raw_utts if len(utt.split("GEN (default): ")) > 1]

    #* T5 NEW
    # print("T5 (NEW)")
    # utt_path = r"C:\Users\Leo\Desktop\test_predictions.json"
    # with open(utt_path, "r", encoding="utf-8") as f:
    #     raw_utts = json.load(f)
    # utterances = [utt["gen"][0] for utt in raw_utts]
    
    #* BASE
    print("BASE")
    utterances = [entry["text"][-1] for entry in dataset]
    #* T5 BASE CE SER ES
    # print("T5 (BASE CE SER ES)")
    # utt_path = r"C:\Users\Leo\Desktop\webnlg_base_ce_ser_es_predictions.json"
    # with open(utt_path, "r", encoding="utf-8") as f:
    #     raw_utts = json.load(f)
    # utterances = [utt["gen"][0] for utt in raw_utts]
    
    outputs = calculate_ser(mrs, utterances, base_dataset_path)
    # outputs = calculate_webnlg_ser(mrs, utterances, base_dataset_path, loose_tokenized_search=True, datatuner_format=True)
    print(f"SER: {outputs[0]*100:.3f}")
    print(f"UER: {outputs[2]*100:.3f}")
