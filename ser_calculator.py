import json
import os
from typing import List

from constants import SlotNameConversionMode
from slot_aligner.data_analysis import count_errors
from data_loader import MRToTextDataset
from dataset_loaders.e2e import E2ECleanedDataset
from dataset_loaders.viggo import ViggoDataset
import webnlg_ser_extractor, jilda_ser_extractor, webnlg_ser_extractor_original


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
                    save_errors: bool = False) -> float:
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
        print(f"{base_dataset_path=}")
        outputs = webnlg_ser_extractor.calculate_webnlg_ser(mrs_raw, utterances, base_dataset_path, save_errors=save_errors)
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

    #* Count the missing and hallucinated slots in the utterances
    error_counts = []
    total_content_slots = 0
    for mr_as_list, utt in zip(mrs_processed, utterances):
        num_errors, _, _, num_content_slots = count_errors(
            utt, mr_as_list, dataset_class.name
        )
        error_counts.append(num_errors)
        total_content_slots += num_content_slots
    #* create a file with all the entries containing errors
    if save_errors:
        total_errors = []
        for i, errors in enumerate(error_counts):
            if errors == 0:
                continue
            total_errors.append({"mr": mrs_raw[i], "hyp": utterances[i], "errors": errors})
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
