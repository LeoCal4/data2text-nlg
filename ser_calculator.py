import json
import os
from typing import List

from constants import SlotNameConversionMode
from slot_aligner.data_analysis import count_errors
from data_loader import MRToTextDataset
from dataset_loaders.e2e import E2ECleanedDataset
from dataset_loaders.viggo import ViggoDataset
import webnlg_ser_extractor, jilda_ser_extractor


def get_dataset_class(dataset_name: str) -> MRToTextDataset:
    if "viggo" in dataset_name: # to include viggo-small
        dataset_class = ViggoDataset
    elif dataset_name == "e2e":
        dataset_class = E2ECleanedDataset
    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name")
    return dataset_class


def calculate_ser(mrs_raw: List[str], utterances: List[str], dataset_name: str, 
                    output_uer: bool = False, base_dataset_path: str = None,
                    save_errors: bool = False, datatuner_format: bool = False) -> float:
    """Analyzes unrealized and hallucinated slot mentions in the utterances."""
    #* check if the dataset is webnlg
    # TODO clean repeated code
    if "jilda" in dataset_name:
        if base_dataset_path is None:
            # should be /something/datatuner/src/datatuner/lm/custom/libs/data2text-nlp, so we remove everything after the first datatuner folder
            datatuner_folder = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-6])
            base_dataset_path = os.path.join(datatuner_folder, "data", dataset_name)
        outputs = jilda_ser_extractor.calculate_ser(mrs_raw, utterances, base_dataset_path, save_errors=save_errors)
        if not output_uer:
            outputs = outputs[:2]
        return outputs
    elif "webnlg" in dataset_name:
        if base_dataset_path is None:
            # should be /something/datatuner/src/datatuner/lm/custom/libs/data2text-nlp, so we remove everything after the first datatuner folder
            datatuner_folder = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-6])
            base_dataset_path = os.path.join(datatuner_folder, "data", dataset_name)
        outputs = webnlg_ser_extractor.calculate_webnlg_ser(mrs_raw, utterances, base_dataset_path, save_errors=save_errors, datatuner_format=datatuner_format)
        # outputs = webnlg_ser_extractor_original.calculate_ser(mrs_raw, utterances, base_dataset_path)
        if not output_uer:
            outputs = outputs[:2]
        return outputs

    #* Load MRs and corresponding utterances
    try:
        dataset_class = get_dataset_class(dataset_name)
    except ValueError:
        print(f"ERROR: could not provide SER for dataset {dataset_name}")
        outputs = -1.0, -1.0
        if output_uer:
            outputs += -1.0, -1.0
        return outputs
    mrs_processed = dataset_class.preprocess_mrs(mrs_raw, as_lists=True, lowercase=False, slot_name_conversion=SlotNameConversionMode.SPECIAL_TOKENS)

    #* Count the missing, hallucinated and replicated slots in the utterances, 
    error_counts = [] # include replication errors
    replication_errors_counts = []
    total_content_slots = 0
    for mr_as_list, utt in zip(mrs_processed, utterances):
        num_errors, _, replication_errors, num_content_slots = count_errors(
            utt, mr_as_list, dataset_class.name
        )
        replication_errors_counts.append(len(replication_errors))
        error_counts.append(num_errors)
        total_content_slots += num_content_slots
    #* create a file with all the entries containing errors
    if save_errors:
        total_errors = []
        for mr, utt, errors, repl_errors in zip(mrs_raw, utterances, error_counts, replication_errors_counts):
            if errors == 0:
                continue
            total_errors.append({"mr": mr, "hyp": utt, "errors": errors, "repl": repl_errors})
        with open(f"{dataset_name}_errors.json", "w", encoding="utf-8") as f:
            json.dump(total_errors, f, ensure_ascii=False, sort_keys=False, indent=4)
    #* Calculate SER
    ser = sum(error_counts) / total_content_slots
    outputs = ser, sum(error_counts)
    if output_uer:
        wrong_sentences = sum([num_errs > 0 for num_errs in error_counts])
        uer = wrong_sentences / len(utterances)
        outputs += uer, wrong_sentences
    return outputs


def extract_original_datatuner_refs(datatuner_refs_path: str) -> List[str]:
    refs = []
    with open(datatuner_refs_path, "r") as f:
        datatuner_json = json.load(f)
        for entry in datatuner_json:
            refs.append(entry["ref         "][0])
    return refs


if __name__ == "__main__":
    dataset_name = "e2e"
    base_dataset_path = fr"C:\Users\Leo\Documents\PythonProjects\Tesi\datatuner\data\{dataset_name}"
    dataset =  []
    partitions = ["test"]
    for partition in partitions:
        with open(os.path.join(base_dataset_path, f"{partition}.json"), "r", encoding="utf-8") as f:
            dataset.extend(json.load(f))
    mrs = [entry["mr"] for entry in dataset]

    print(f"DT - {dataset_name}")
    folder_name = "base" if dataset_name != "viggo" else "og"
    utt_path = fr"C:\Users\Leo\Desktop\Tesi\results\{dataset_name}\{dataset_name}_{folder_name}_results\reranked.json"
    # if dataset_name == "viggo":
    #     utt_path += r"\generated.json"
    # elif dataset_name == "e2e":
    #     utt_path += r"\reranked.json"
    utterances = extract_original_datatuner_refs(utt_path)

    assert len(mrs) == len(utterances)
    outputs = calculate_ser(mrs, utterances, dataset_name, output_uer=True)
    print(f"SER: {outputs[0]*100:.3f} ({outputs[1]} slots)")
    print(f"UER: {outputs[2]*100:.3f} ({outputs[3]} sentences)")
