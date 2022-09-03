#   descrizione_lavoro -> troppo vago quando è nessuno/?
#   contatto -> ha sempre un valore
#   nome_azienda -> ha sempre un valore


from typing import List, Union
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
             '_': ' ', '"': '', "–": "-", "\\": "", '’': "'"}


#* written without last letter to get both singulars and plurals
SYNONYMS = {
    "<doveri>": ["dover", "mansion"],
    "<settore>": ["settor", "lavor", "ambit", "profilo", "materi", "campo", "campi", "area"],
    "<abilità>": ["competenz", "abilit", "skill", "conosc", "aggettiv", "esperienz", "caratter", "capacita", "capacità", "doti"],
    "<titolo_di_studio>": ["studi", "studente", "formazion", "certificazion", "laurea", "diploma", "competenz", "facolta", "percorso formativo", "percorso accademico"],
    "<grandezza_azienda>": ["grandezza", "dimension", "grand", "piccol", "media", "medio", "tipo di azienda"],
    "<luogo>": ["luog", "città", "citta", "region", "stato", "provincia", "trasferi", "geografic", "spostar", "rimanere", "dove", "zon", "ubicazion", "italia", "estero"],
    "<contratto>": ["contratt", "posizione"],
    "<esperienze_lavorative>": ["esperienz", "lavorat", "passato"],
    "<lingue>": ["competenz", "abilit", "skill", "conoscenz", "lingu", "italiano"],
    "<contatto>": ["contatt", "riferimento"],
    "<età>": ["anagrafic", "anni", "eta", "età"],
    "<descrizione_lavoro>": ["impiego", "offerta", "lavoro", "opportunit"] # in case of ?
}
NEGATIONS = ["no", "non", "purtroppo", "nessuna", "nessuno", "nessun", "nemmeno"]
PLACEHOLDER_VALUES = ["?", "nessuno", "none"]
ERRORS = []



def clean_entity(entity: str) -> str:
    entity = entity.lower().replace('_', ' ')
    # separate punct signs from text
    entity = ' '.join(re.split('(\W)', entity))
    entity = ' '.join(entity.split())  # delete whitespaces
    return entity


def delexicalize_slot_value(slot_value: str) -> str:
    translation = slot_value.maketrans(CHARACTER_NORMALIZATION_MAP)
    return slot_value.translate(translation)


def extract_slot_values_from_dialogue_act(dialogue_act: str) -> List[str]:
    slot_values = re.findall(r"[\w\s]*=([^\|\.]*)", dialogue_act)
    return [slot_value.strip() for slot_value in slot_values]


def get_all_slot_values(base_dataset_path: str):
    #* read the dataset
    complete_dataset = []
    for partition in ["train", "validation", "test"]:
        partition_path = os.path.join(base_dataset_path, f"{partition}.json")
        with open(partition_path, "r", encoding="utf-8") as f:
            complete_dataset += json.load(f)
    complete_mrs = [entry["mr"].lower() for entry in complete_dataset]
    slot_values = []
    #* extract and clean all slot values
    for mr in complete_mrs: # mr = meaning representation aka the data
        dialogue_acts = mr.split(" . ")
        for da in dialogue_acts:
            if "<altro>" in da:
                continue
            slot_values.extend(extract_slot_values_from_dialogue_act(da))
    slot_values = list(set(slot_values))
    #* these values are too ambigous and show up in way too many entries to be reliable hallucinations
    HALLUCINATION_AMBIGUOUS_WORDS = [
        "annuncio",
        "non è specificato", # contratto, luogo
        "ricerca", # doveri
        "precisa", # abilità
        "proposta", # descrizione lavoro
        "esperienza", # esperienze_lavorative
        "provincia", # luogo
        "controllo", # doveri
        "una persona", # altro
        "progetto", 
        "durata", # 
    ]
    [slot_values.remove(ambiguous_word) for ambiguous_word in HALLUCINATION_AMBIGUOUS_WORDS if ambiguous_word in slot_values]
    return slot_values


def count_missing_values(slot_types: List[str], slot_values: List[str], cleaned_prediction: str, cleaned_mr: str) -> int:
    global ERRORS
    n_missing = 0
    #* VALUES:
    #* - ANY non special VALUE: check verbatim
    #* - ?: do nothing (explanation below)
    #* - nessuno: check negation
    #! all slot with ? value are part of <richiesta> da, while ~95% of richiesta DAs has a ? in a slot value
    #!  for this reason, we don't need to check for questions hints when the value is ?, we just ignore it 
    for slot_type, slot_value in zip(slot_types, slot_values):
        #* nothing we can do about this case, so we just count it as wrong
        if slot_type == "<altro>" and slot_value in PLACEHOLDER_VALUES:
            n_missing += 1
            ERRORS.append({"error_type": "CURSED", "slot_type": slot_type, "slot_value": slot_value, "mr": cleaned_mr, "prediction": cleaned_prediction})
            continue
        #* mostly just the same as a negation da
        if slot_type == "<nessuno>" or slot_value == "nessuno":
            #* negations are checked on token level to be sure that it is not just a part of another word
            is_negation_present = any([negation in cleaned_prediction.split() for negation in NEGATIONS])
            if not is_negation_present:
                n_missing += 1
                ERRORS.append({"error_type": "NESSUNO - SLOT TYPE", "mr": cleaned_mr, "prediction": cleaned_prediction})
        #* if slot_value is a placeholder value, check for the presence of the slot type or its synonyms
        if slot_type != "<nessuno>" and slot_value in PLACEHOLDER_VALUES:
            slot_type_synonyms = SYNONYMS[slot_type]
            is_slot_type_present = any([slot_type_syn in cleaned_prediction for slot_type_syn in slot_type_synonyms])
            if not is_slot_type_present:
                n_missing += 1
                ERRORS.append({"error_type": "SLOT_TYPE", "slot_type_synonyms": slot_type_synonyms, "mr": cleaned_mr, "prediction": cleaned_prediction})
        #* if slot_value is an actual value, check for the presence of that value
        else:
            cleaned_slot_value = clean_entity(slot_value)
            delixicalized_slot_value = delexicalize_slot_value(cleaned_slot_value)
            if cleaned_slot_value not in cleaned_prediction and delixicalized_slot_value not in cleaned_prediction:
                n_missing += 1
                ERRORS.append({"error_type": "REAL VALUES", "slot_value": slot_value, "mr": cleaned_mr, "prediction": cleaned_prediction})
    return n_missing


def is_substring_strictly_present(main: Union[str, List[str]], sub: Union[str, List[str]]) -> bool:
    #* sub is checked in main in order
    if type(main) is not list:
        main = main.split()
    if type(sub) is not list:
        sub = sub.split()
    previous_index = None
    for token in sub:
        try:
            current_index = main.index(token)
            #* check if substring is present maintaining its original order
            if previous_index and previous_index != current_index-1:
                return False
            previous_index = current_index
        #* substring token is not present 
        except ValueError:
            return False
    return True

#! trying with main as string
def is_substring_loosely_present(main: str, sub: Union[str, List[str]]) -> bool:
# def is_substring_loosely_present(main: Union[str, List[str]], sub: Union[str, List[str]]) -> bool:
    #* check if at least half of sub's tokens are present in main
    # if type(main) is not list:
    #     main = main.split()
    if type(sub) is not list:
        sub = sub.split()
    # print(f"{[token[:-1] for token in sub if len(token) > 3]}")
    # result = (sum([token in main for token in sub]) / len(sub)) >= 0.5
    # result = (sum([token[:-1] in main if len(token) > 3 else token in main for token in sub]) / len(sub)) >= 0.5
    # if not result:
    #     print(f"{main=}") 
    #     print(f"{sub=}") 
    #     print("===========================")
    # return result
    return (sum([token[:-1] in main if len(token) > 3 else token in main for token in sub]) / len(sub)) >= 0.5
    # return (sum([token in main for token in sub]) / len(sub)) >= 0.5


def calculate_ser(mrs_raw: List[str], utterances: List[str], base_dataset_path: str, save_errors: bool = False):
    global ERRORS
    ERRORS = []
    assert len(mrs_raw) == len(utterances)
    #* remove the prefix and <data> token if present
    mrs_raw = [mr.split("<data>")[1].strip() if "<data>" in mr else mr for mr in mrs_raw]
    mrs_raw = [mr.split(". <text>")[0].strip() if ". <text>" in mr else mr for mr in mrs_raw]
    # general structure: <DA> DA | [<SLOT> SLOT = VALUE]+ [. <DA> DA | ]*
    # decisions need to be taken first on DA level then on SLOT level
    #* get all entities from the whole corpus
    all_slot_values = get_all_slot_values(base_dataset_path)
    wrong_entries = []
    total_n_slots = 0
    wrong_slots_per_entry = []
    for meaning_representation, prediction in zip(mrs_raw, utterances):
        n_missing = 0
        n_hallucinated = 0
        n_repeated = 0
        #* divide the data into the dialogue acts composing it
        dialogue_acts = meaning_representation.split(" . ")
        cleaned_prediction = clean_entity(prediction)
        cleaned_mr = clean_entity(meaning_representation)
        repeated_entries = []
        for dialogue_act in dialogue_acts:
            if not dialogue_act:
                continue
            #* get da, slot types and names
            slot_names = re.findall(r"<[\w\_]*>", dialogue_act)
            dialogue_act_type = slot_names[0].strip()
            slot_types = [slot_name.strip() for slot_name in slot_names[1:]]
            slot_values = extract_slot_values_from_dialogue_act(dialogue_act)
            #* check REPETITIONS
            for slot_value in slot_values:
                if slot_value in PLACEHOLDER_VALUES or len(slot_value) <= 1:
                    continue
                cleaned_slot_value = clean_entity(slot_value)
                reps_count = cleaned_prediction.count(cleaned_slot_value)
                og_reps_count = cleaned_mr.count(cleaned_slot_value)
                if reps_count > og_reps_count and reps_count > 2:
                    repeated_entries.append(slot_value)
                    n_repeated += reps_count - 1
            #* decide what to do based on da type
            if dialogue_act_type == "<richiesta>":
                REQUESTS = ["?", "chied", "raccont", "dimmi", "parlami"]
                is_request_present = any([request in cleaned_prediction for request in REQUESTS])
                if not is_request_present:
                    n_missing += 1
                    ERRORS.append({"mr": dialogue_act, "error_type": "RICHIESTA", "prediction": cleaned_prediction})
                #* check for missing values
                n_missing += count_missing_values(slot_types, slot_values, cleaned_prediction, cleaned_mr)
            #* check if there is a negation in the sentence + verbatim check of slot values
            elif dialogue_act_type == "<rifiuta>":
                #* negations are checked on token level to be sure that it is not just a part of another word
                is_negation_present = any([negation in cleaned_prediction.split() for negation in NEGATIONS])
                if not is_negation_present:
                    n_missing += 1
                    ERRORS.append({"mr": dialogue_act, "error_type": "RIFIUTA", "prediction": cleaned_prediction})
                #* check for missing values
                n_missing += count_missing_values(slot_types, slot_values, cleaned_prediction, cleaned_mr)
            #* only check for missing values
            #! <seleziona> is vary vague so we just check for values
            elif dialogue_act_type == "<informa>" or dialogue_act_type == "<seleziona>":
                n_missing += count_missing_values(slot_types, slot_values, cleaned_prediction, cleaned_mr)
            else:
                print(f"ERROR: dialogue_act_type {dialogue_act_type} not recognized")
                continue
            #* check for HALLUCINATIONS
            current_slot_names_synonyms = []
            [current_slot_names_synonyms.extend(SYNONYMS[slot_name]) for slot_name in slot_names if slot_name in SYNONYMS]
            current_slot_names_synonyms = " ".join(current_slot_names_synonyms)
            all_slot_values_not_present = [item for item in all_slot_values if item not in slot_values]
            hallucinated_entities = []
            tokenized_prediction = cleaned_prediction.split()
            for entity in all_slot_values_not_present:
                cleaned_entity = clean_entity(entity)
                delex_entity = delexicalize_slot_value(cleaned_entity)
                if len(entity) <= 3:
                    continue
                #* check for both the cleaned and the delexicalized entities:
                #*  - their tokenized versions are present in the tokenized prediction (strictly and in order) 
                #*  - they are not present in the cleaned meaning prepresentation
                #*  - their versions without the last letter are not present in the slot names synonyms (issue with )
                if (is_substring_strictly_present(tokenized_prediction, cleaned_entity) or is_substring_strictly_present(tokenized_prediction, delex_entity)) and \
                    (not is_substring_loosely_present(cleaned_mr, cleaned_entity) and not is_substring_loosely_present(cleaned_mr, delex_entity)) and \
                    (cleaned_entity[:-1] not in current_slot_names_synonyms and delex_entity[:-1] not in current_slot_names_synonyms):
                    #* avoid repeating partial entities e.g. "guida" when "guida turistica" has already been added
                    all_hallucinated_entities = " ".join(hallucinated_entities)
                    if entity in all_hallucinated_entities:
                        continue
                    hallucinated_entities.append(entity)
                    n_hallucinated += 1
            if n_missing > 0 or n_hallucinated > 0 or n_repeated > 0:
                wrong_entries.append({
                    "mr": meaning_representation, 
                    "pred": prediction, 
                    # "missing": total_missing_data, 
                    "n_missing": n_missing, 
                    # "entities": total_wrong_entities, 
                    "n_hallucinated": n_hallucinated, 
                    "hallucinated": hallucinated_entities,
                    "n_repetition": n_repeated,
                    "repeated": repeated_entries,
                })
            wrong_slots_per_entry.append(n_missing + n_hallucinated + n_repeated)
            total_n_slots += len(slot_names)
    if save_errors:
        with open("jilda_ser_results.json", "w", encoding="utf-8") as f:
            json.dump(wrong_entries, f, ensure_ascii=False, indent=4, sort_keys=False)
        with open("jilda_ser_errors.json", "w", encoding="utf-8") as f:
            json.dump(ERRORS, f, ensure_ascii=False, indent=4, sort_keys=False)
    n_wrong_slots = sum(wrong_slots_per_entry)
    ser = n_wrong_slots / total_n_slots
    n_wrong_sentences = sum([num_errs > 0 for num_errs in wrong_slots_per_entry])
    uer = n_wrong_sentences / len(wrong_slots_per_entry) # utterance error rate
    return ser, n_wrong_slots, uer, n_wrong_sentences



if __name__ == "__main__":
    base_dataset_path = r"C:\Users\Leo\Documents\PythonProjects\Tesi\datatuner\data\jilda"
    dataset = []
    # partitions = ["test"]
    partitions = ["train", "validation", "test"]
    for partition in partitions:
        with open(os.path.join(base_dataset_path, f"{partition}.json"), "r", encoding="utf-8") as f:
            dataset.extend(json.load(f))
    mrs = [entry["mr"] for entry in dataset]

    #* T5 NEW
    # print("T5 (NEW)")
    # utt_path = r"C:\Users\Leo\Desktop\jilda_predictions.json"
    # with open(utt_path, "r", encoding="utf-8") as f:
    #     raw_utts = json.load(f)
    # utterances = [utt["gen"][0] for utt in raw_utts]
    
    #* BASE
    print("BASE")
    utterances = [entry["ref"] for entry in dataset]

    outputs = calculate_ser(mrs, utterances, base_dataset_path, save_errors=True)
    print(f"SER: {outputs[0]*100:.3f} ({outputs[1]} slots)")
    print(f"UER: {outputs[2]*100:.3f} ({outputs[3]} sentences)")

