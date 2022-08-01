from typing import List, Union, Tuple
import re
import os
import json


CHARACTER_NORMALIZATION_MAP = {'À': 'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
             'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'ª': 'A', 'ă': 'a',
             'È': 'E', 'É': 'E', 'Ê': 'E', 'Ë': 'E',
             'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
             'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
             'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i', 'ı': 'i',
             'Ò': 'O', 'Ó': 'O', 'Ô': 'O', 'Õ': 'O', 'Ö': 'O',
             'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'º': 'O', 'ø': 'o', 'ō': 'o',
             'Ù': 'U', 'Ú': 'U', 'Û': 'U', 'Ü': 'U',
             'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u', 'ü': 'u',
             'Ñ': 'N', 'ñ': 'n',
             'Ç': 'C', 'ç': 'c',
             'ș': 's', 'ş': 's',
             'ğ': 'g',
             '§': 'S',  '³': '3', '²': '2', '¹': '1',
             '_': ' ', '"': '', "–": "-", "\\": ""}


MONTHS = {
    "01": "january",
    "02": "february",
    "03": "march",
    "04": "april",
    "05": "may",
    "06": "june",
    "07": "july",
    "08": "august",
    "09": "september",
    "10": "october",
    "11": "november",
    "12": "december"
}

WORDS_ALTERNATIVES = {
    "england": ["united kingdom", "uk", "u.k.", "english"],
    "united kingdom": ["uk", "u.k.", "england", "english"],
    "united states": ["us", "u.s.", "america", "u.s.a.", "usa"],
    "us": ["united states", "u.s.", "america", "u.s.a.", "usa"],
    "uk": ["united kingdom", "english", "england", "u.k."],
    "italy": ["italian"],
    "italian": ["italy"],
    "world war ii": ["wwii", "ww2"],
    "world war i": ["wwi", "ww1"],
    "ireland": ["irish"],
    "germany": ["german"]
}


def normalize_entity_chars(entity: Union[str, List]) -> str:
    if type(entity) is str: 
        translation = entity.maketrans(CHARACTER_NORMALIZATION_MAP)
        return entity.translate(translation)
    elif type(entity) is list:
        return [normalize_entity_chars(subentity) for subentity in entity]
    else:
        raise


def clean_webnlg_entity(entity: str) -> str:
    entity = entity.lower().strip().replace('_', ' ')
    #* remove values between parentheses
    entity = re.sub(r"\([\w\d]*\)", "", entity)
    #* delete whitespaces
    entity = ' '.join(entity.split())
    #* remove -ing from verbs
    entity = entity[:-3] if entity.endswith("ing") else entity
    return entity


def get_all_subj_obj(base_dataset_path: str):
    #* read all the webnlg corpus
    complete_dataset = []
    for partition in ["train", "validation", "test"]:
        partition_path = os.path.join(base_dataset_path, f"{partition}.json")
        with open(partition_path, "r", encoding="utf-8") as f:
            complete_dataset += json.load(f)
    # print(f"{len(complete_dataset)=}")
    complete_mrs = [entry["modifiedtripleset"] for entry in complete_dataset]
    subjects = []
    objects = []
    #* extract all subjects and objects
    for mr in complete_mrs: # mr = meaning representation aka the data
        # separated_mrs = mr.split(" ||| ")
        separated_mrs = mr.split(" ; ")
        for sep_mr in separated_mrs:
            subject, obj = extract_subjects_and_objects_from_mr(sep_mr, datatuner_format=True)
            subjects.append(subject)
            objects.append(obj)
    #* clean subj and obj
    subjs_cleaned = []
    for subj in list(subjects):
        subjs_cleaned.append(clean_webnlg_entity(subj))
    objs_cleaned = []
    for obj in list(objects):
        objs_cleaned.append(clean_webnlg_entity(obj))
    return subjs_cleaned, objs_cleaned


def extract_subjects_and_objects_from_mr(mr: str, datatuner_format: bool = False) -> List[str]:
    #* extract all subjects and objects and clean them
    subj_obj = []
    mr = mr.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    mr = re.sub(r"\([\w\d]*\)", "", mr)
    #* check for dates
    if datatuner_format:
        separated_mrs = mr.split(" ; ")
    else:
        separated_mrs = mr.split(" ||| ")
    for sep_mr in separated_mrs:
        if sep_mr == "":
            continue
        try:
            if datatuner_format:
                subject, _, obj = re.findall(r"<[\w\s]*>\s*([^<;]*)(?:;)?", sep_mr.strip())
            else:
                subject, _, obj = sep_mr.strip().split(" | ")
        except:
            print(sep_mr)
            return [""]
        subject = clean_webnlg_entity(subject)
        obj = clean_webnlg_entity(obj)
        #* delete white spaces
        subj_obj.append(' '.join(subject.split()))
        subj_obj.append(' '.join(obj.split()))
    return subj_obj


def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def parse_date(date_data: re.Match) -> str:
    #* year
    entries = date_data.group(1),
    #* month
    entries += (MONTHS[date_data.group(2)], date_data.group(2))
    #* days
    entries += date_data.group(3),
    if date_data.group(3)[0] == '0':
        entries += date_data.group(3)[1],
    return entries


def expand_entries(entry: str) -> Tuple[List[str], str]:
    entries = [entry, normalize_entity_chars(entry)]
    prefix = "OTHER"
    #* check if value is a date
    date_data = re.match(r"(\d{4})-(\d{2})-(\d{2})", entry)
    if date_data:
        entries = parse_date(date_data)
        prefix = "DATE"
    #* check if value has -
    elif "-" in entry and not is_float(entry):
        entries += entry.replace("-", " "), entry.replace("-", "")
        prefix = "HYPHEN"
    #* check with and without parentheses
    if "(" in entry or ")" in entry:
        entries += [parsed_entry.replace("(", "").replace(")", "") for parsed_entry in entries]
        prefix = "PARENTHESES"
    #* check number as it is, as a float and as an int (without decimals)
    if is_float(entry):
        base_entries = entries
        entries += [str(float(parsed_entry)) for parsed_entry in base_entries]
        entries += [str(int(float(parsed_entry))) for parsed_entry in base_entries]
        prefix = "NUMERICAL"
    #* check with and without punctuation, and with spaces instead of punctuation
    elif "." in entry or "," in entry:
        base_entries = entries
        entries += [parsed_entry.replace(".", "").replace(",", "") for parsed_entry in base_entries]
        entries += [parsed_entry.replace(".", " ").replace(",", " ") for parsed_entry in base_entries]
        entries += [parsed_entry.replace(".", ". ").replace(",", ", ") for parsed_entry in base_entries]
        prefix = "PUNCTUATION"
    if "&" in entry:
        entries += [parsed_entry.replace("&", "and") for parsed_entry in entries]
        prefix = "&"
    if entry[-1] == 's':
        entries += [parsed_entry[:-1] for parsed_entry in entries if parsed_entry[-1] == 's']
        prefix = "PLURAL"
    return entries, prefix


def remove_word_boundaries(sentence: str) -> str:
    return sentence.replace("(", " ").replace(")", " ").replace(".", " ").replace(",", " ")


def calculate_webnlg_ser(mrs_raw: List[str], utterances: List[str], base_dataset_path: str, 
                            loose_tokenized_search: bool = False, datatuner_format: bool = False):
    #* get all cleaned subjects and objects from the whole corpus
    all_subj_cleaned, all_obj_cleaned = get_all_subj_obj(base_dataset_path)
    all_entities = list(set(all_subj_cleaned + all_obj_cleaned))
    #* delete all numbers from entities
    for i, entity in enumerate(all_entities):
        try:
            float(entity)
            del all_entities[i]
        except ValueError:
            pass
    wrong_entries = []
    wrong_slots_per_entry = []
    total_n_slots = 0
    for mr, pred in zip(mrs_raw, utterances):
        pred = pred.lower()
        pred = normalize_entity_chars(pred)
        mr = mr.lower()
        mr = normalize_entity_chars(mr)
        subjs_and_objs = extract_subjects_and_objects_from_mr(mr, datatuner_format=datatuner_format)
        n_slots = len(subjs_and_objs)
        missing_entities = 0
        n_hallucinated_entities = 0
        total_missing_data = []
        total_wrong_entities = []
        #* cycle subjects and objects
        for entity in subjs_and_objs:
            current_missing_data = []
            current_wrong_entities = []
            expanded_entities = [entity] + WORDS_ALTERNATIVES.get(entity, [])
            expanded_entities_absence = []
            for value in expanded_entities:
                missing_data = None
                if not loose_tokenized_search:
                    if value not in pred and normalize_entity_chars(value) not in pred:
                        missing_data = value
                else:
                    value = value.split()
                    for entry in value:
                        entries, prefix = expand_entries(entry)
                        #* check if it actually was some missing data
                        entries_absence = [item not in pred for item in entries]
                        if entries_absence and all(entries_absence):
                            missing_data = f"{prefix}: "  + " | ".join(entries)
                            current_wrong_entities.extend(entries)
                            current_missing_data.append(missing_data)
                            break
                expanded_entities_absence.append(missing_data is not None)
            expanded_entities_missing = int(all(expanded_entities_absence))
            missing_entities += expanded_entities_missing
            if expanded_entities_missing:
                total_missing_data.extend(current_missing_data)
                total_wrong_entities.extend(current_wrong_entities)
        #* delete s and o that are present in MR
        all_subj_and_obj_not_present = [item for item in all_entities if item not in subjs_and_objs]
        hallucinated_entities = []
        for entity in all_subj_and_obj_not_present:
            if len(entity) < 2:
                continue
            if f" {entity} " in pred and not f"{remove_word_boundaries(entity)}" in remove_word_boundaries(mr):
                hallucinated_entities.append(entity)
                n_hallucinated_entities += 1
        total_n_slots += n_slots
        wrong_slots_per_entry.append(missing_entities + n_hallucinated_entities)
        if missing_entities > 0 or n_hallucinated_entities > 0:
            wrong_entries.append({
                "mr": mr, 
                "pred": pred, 
                "missing": total_missing_data, 
                "n_missing": missing_entities, 
                "entities": total_wrong_entities, 
                "n_hallucinated": n_hallucinated_entities, 
                "hallucinated": hallucinated_entities
            })
    total_wrong_slots = sum(wrong_slots_per_entry)
    ser = (total_wrong_slots / total_n_slots) # slot error rate
    wrong_sentences = sum([num_errs > 0 for num_errs in wrong_slots_per_entry])
    uer = (wrong_sentences / len(wrong_slots_per_entry)) # utterance error rate
    with open("ser_results.json", "w", encoding="utf-8") as f:
        json.dump(wrong_entries, f, ensure_ascii=False, indent=4, sort_keys=False)
    return ser, total_wrong_slots, uer, wrong_sentences


def extract_original_datatuner_refs(datatuner_refs_path: str) -> List[str]:
    refs = []
    with open(datatuner_refs_path, "r") as f:
        datatuner_json = json.load(f)
        for entry in datatuner_json:
            try:
                zipped_entry = list(zip(entry["pred"], entry["pred_prob"], entry["reranked"]))
                sorted_entry = sorted(zipped_entry, key=lambda x: (x[0], -x[1])) # pred, -pred_prob
                refs.append(sorted_entry[0][2])
            except:
                refs.append(entry["text         "][0])
    return refs



if __name__ == "__main__":
    base_dataset_path = r"C:\Users\Leo\Documents\PythonProjects\Tesi\datatuner\data\webnlg"
    with open(os.path.join(base_dataset_path, "test.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)
    mrs = [entry["modifiedtripleset"] for entry in dataset]
    
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
    
    outputs = calculate_webnlg_ser(mrs, utterances, base_dataset_path, loose_tokenized_search=True, datatuner_format=True)
    print(f"SER: {outputs[0]*100:.3f} ({outputs[1]} slots)")
    print(f"UER: {outputs[2]*100:.3f} ({outputs[3]} sentences)")
