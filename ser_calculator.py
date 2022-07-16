from typing import List

from constants import SlotNameConversionMode
from slot_aligner.data_analysis import count_errors
from data_loader import MRToTextDataset
from dataset_loaders.e2e import E2ECleanedDataset
from dataset_loaders.viggo import ViggoDataset


def get_dataset_class(dataset_name: str) -> MRToTextDataset:
    if "viggo" in dataset_name: # to include viggo-small
        dataset_class = ViggoDataset
    elif dataset_name == "e2e":
        dataset_class = E2ECleanedDataset
    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name")
    return dataset_class


def calculate_ser(mrs_raw: List[str], utterances: List[str], dataset_name: str, output_uer: bool = False) -> float:
    """Analyzes unrealized and hallucinated slot mentions in the utterances."""
    #* Load MRs and corresponding utterances
    dataset_class = get_dataset_class(dataset_name)
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

    #* Calculate SER
    ser = sum(error_counts) / total_content_slots
    outputs = ser, sum(error_counts)
    if output_uer:
        wrong_sentences = sum([num_errs > 0 for num_errs in error_counts])
        uer = wrong_sentences / len(utterances)
        outputs += uer, wrong_sentences
    return outputs
