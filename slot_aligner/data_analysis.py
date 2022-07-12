import argparse
import json
import os
from collections import Counter, OrderedDict

import pandas as pd
from data_loader import MRToTextDataset
from dataset_loaders.e2e import E2ECleanedDataset
from dataset_loaders.viggo import ViggoDataset
from slot_aligner.slot_alignment import count_errors, find_alignment
from typing import List


def align_slots(data_dir, filename, dataset_class, serialize_pos_info=False):
    """Finds and records the position of each slot's mention in the corresponding utterance.

    The position is indicated as the index of the first character of the slot mention phrase within the utterance. When
    the phrase comprises non-contiguous words in the utterance, the position is typically that of the salient term in
    the phrase.

    Note that if a slot is mentioned in the corresponding utterance multiple times, only its last mention is recorded.
    """
    alignments = []

    # Load MRs and corresponding utterances
    df_data = pd.read_csv(os.path.join(data_dir, filename), header=0)
    mrs_raw = df_data.iloc[:, 0].to_list()
    mrs_processed = dataset_class.preprocess_mrs(mrs_raw, as_lists=True, lowercase=False, convert_slot_names=True)
    utterances = df_data.iloc[:, 1].to_list()

    for mr_as_list, utt in zip(mrs_processed, utterances):
        # Determine the positions of all slot mentions in the utterance
        slot_mention_positions = find_alignment(utt, mr_as_list, dataset_class.name)
        if serialize_pos_info:
            alignments.append(json.dumps([[pos, slot] for slot, _, pos in slot_mention_positions]))
        else:
            alignments.append(' '.join([f'({pos}: {slot})' for slot, _, pos in slot_mention_positions]))

    # Save the MRs and utterances along with the positional information about slot mentions to a new CSV file
    df_data['alignment'] = alignments
    out_file_path = os.path.splitext(os.path.join(data_dir, filename))[0] + ' [with alignment].csv'
    df_data.to_csv(out_file_path, index=False, encoding='utf-8-sig')


def score_slot_realizations(data_dir, predictions_file, dataset_class, slot_level=False, verbose=False,
                             output_dir_path: str = None, output_name: str = "test"):
    """Analyzes unrealized and hallucinated slot mentions in the utterances."""

    error_counts = []
    incorrect_slots = []
    duplicate_slots = []
    total_content_slots = 0

    # Load MRs and corresponding utterances
    df_data = pd.read_csv(os.path.join(data_dir, predictions_file), header=0)
    mrs_raw = df_data.iloc[:, 0].to_list()
    mrs_processed = dataset_class.preprocess_mrs(mrs_raw, as_lists=True, lowercase=False, convert_slot_names=True)
    utterances = df_data.iloc[:, 1].fillna('').to_list()

    for mr_as_list, utt in zip(mrs_processed, utterances):
        # Count the missing and hallucinated slots in the utterance
        num_errors, cur_incorrect_slots, cur_duplicate_slots, num_content_slots = count_errors(
            utt, mr_as_list, dataset_class.name, verbose=verbose)
        error_counts.append(num_errors)
        incorrect_slots.append(', '.join(cur_incorrect_slots))
        duplicate_slots.append(', '.join(cur_duplicate_slots))
        total_content_slots += num_content_slots

    # Save the MRs and utterances along with their slot error indications to a new CSV file
    df_data['errors'] = error_counts
    df_data['incorrect'] = incorrect_slots
    df_data['duplicate'] = duplicate_slots
    if not output_dir_path:
        output_dir_path = data_dir
    out_file_path = os.path.join(output_dir_path, output_name) + '.csv'
    df_data.to_csv(out_file_path, index=False, encoding='utf-8-sig')

    # Calculate the slot-level or utterance-level SER
    if slot_level:
        ser = sum(error_counts) / total_content_slots
    else:
        ser = sum([num_errs > 0 for num_errs in error_counts]) / len(utterances)

    num_wrong_sentences = df_data[df_data['errors'] > 0]["incorrect"].size
    num_tot_sentences = len(mrs_raw)
    percentage_of_wrong_sentences = round(num_wrong_sentences * 100 / num_tot_sentences, 2)

    # Print the SER
    if verbose:
        print(f'>> Slot error rate: {round(100 * ser, 2)}%')
    else:
        print(f"SER:  {round(100 * ser, 2)}% (wrong slots {sum(error_counts)}/{total_content_slots})\n",
              f"SeER: {percentage_of_wrong_sentences}% (wrong sentences: {num_wrong_sentences}/{num_tot_sentences})")

    return ser


def get_dataset_class(dataset_name: str) -> MRToTextDataset:
    if "viggo" in dataset_name: # to include viggo-small
        dataset_class = ViggoDataset
    elif dataset_name == "e2e":
        dataset_class = E2ECleanedDataset
    else:
        raise ValueError(f"{dataset_name} is not a valid dataset name")
    return dataset_class


def calculate_ser(mrs_raw: List[str], utterances: List[str], dataset_name: str) -> float:
    """Analyzes unrealized and hallucinated slot mentions in the utterances."""
    #* Load MRs and corresponding utterances
    dataset_class = get_dataset_class(dataset_name)
    mrs_processed = dataset_class.preprocess_mrs(mrs_raw, as_lists=True, lowercase=False, convert_slot_names=True)

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
    return ser


def score_emphasis(dataset, filename):
    """Determines how many of the indicated emphasis instances are realized in the utterance."""

    emph_missed = []
    emph_total = []

    print('Analyzing emphasis realizations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.EVAL_DIR, dataset, filename))
    dataset_name = data_cont['dataset_name']
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs and the utterances
    mrs = [data_loader.convert_mr_from_str_to_list(mr, data_cont['separators']) for mr in mrs_orig]
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    for i, mr in enumerate(mrs):
        expect_emph = False
        emph_slots = set()
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)

            # Extract slots to be emphasized
            if slot == config.EMPH_TOKEN:
                expect_emph = True
            else:
                mr_dict[slot] = value
                if expect_emph:
                    emph_slots.add(slot)
                    expect_emph = False

        # Delexicalize the MR and the utterance
        utterances[i] = data_loader.delex_sample(mr_dict, utterances[i], dataset=dataset_name)

        # Determine the slot alignment in the utterance
        alignment = find_alignment(utterances[i], mr_dict)

        emph_total.append(len(emph_slots))

        # Check how many emphasized slots were not realized before the name-slot
        for pos, slot, _ in alignment:
            # DEBUG PRINT
            # print(alignment)
            # print(emph_slots)
            # print()

            if slot == 'name':
                break

            if slot in emph_slots:
                emph_slots.remove(slot)

        emph_missed.append(len(emph_slots))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'missed emphasis', 'total emphasis'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['missed emphasis'] = emph_missed
    new_df['total emphasis'] = emph_total

    filename_out = os.path.splitext(filename)[0] + ' [emphasis eval].csv'
    new_df.to_csv(os.path.join(config.EVAL_DIR, dataset, filename_out), index=False, encoding='utf8')


def score_contrast(dataset, filename):
    """Determines whether the indicated contrast relation is correctly realized in the utterance."""

    contrast_connectors = ['but', 'however', 'yet']
    contrast_missed = []
    contrast_incorrectness = []
    contrast_total = []

    print('Analyzing contrast realizations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.EVAL_DIR, dataset, filename))
    dataset_name = data_cont['dataset_name']
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs and the utterances
    mrs = [data_loader.convert_mr_from_str_to_list(mr, data_cont['separators']) for mr in mrs_orig]
    utterances = [data_loader.preprocess_utterance(utt) for utt in utterances_orig]

    for i, mr in enumerate(mrs):
        contrast_found = False
        contrast_correct = False
        contrast_slots = []
        mr_dict = OrderedDict()

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, _, _ = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)

            # Extract slots to be contrasted
            if slot in [config.CONTRAST_TOKEN, config.CONCESSION_TOKEN]:
                contrast_slots.extend(value.split())
            else:
                mr_dict[slot] = value

        # Delexicalize the MR and the utterance
        utterances[i] = data_loader.delex_sample(mr_dict, utterances[i], dataset=dataset_name)

        # Determine the slot alignment in the utterance
        alignment = find_alignment(utterances[i], mr_dict)

        contrast_total.append(1 if len(contrast_slots) > 0 else 0)

        if len(contrast_slots) > 0:
            for contrast_conn in contrast_connectors:
                contrast_pos = utterances[i].find(contrast_conn)
                if contrast_pos < 0:
                    continue

                slot_left_pos = -1
                slot_right_pos = -1
                dist = 0

                contrast_found = True

                # Check whether the correct pair of slots was contrasted
                for pos, slot, _ in alignment:
                    # DEBUG PRINT
                    # print(alignment)
                    # print(contrast_slots)
                    # print()

                    if slot_left_pos > -1:
                        dist += 1

                    if slot in contrast_slots:
                        if slot_left_pos == -1:
                            slot_left_pos = pos
                        else:
                            slot_right_pos = pos
                            break

                if slot_left_pos > -1 and slot_right_pos > -1:
                    if slot_left_pos < contrast_pos < slot_right_pos and dist <= 2:
                        contrast_correct = True
                        break
        else:
            contrast_found = True
            contrast_correct = True

        contrast_missed.append(0 if contrast_found else 1)
        contrast_incorrectness.append(0 if contrast_correct else 1)

    new_df = pd.DataFrame(columns=['mr', 'ref', 'missed contrast', 'incorrect contrast', 'total contrast'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['missed contrast'] = contrast_missed
    new_df['incorrect contrast'] = contrast_incorrectness
    new_df['total contrast'] = contrast_total

    filename_out = os.path.splitext(filename)[0] + ' [contrast eval].csv'
    new_df.to_csv(os.path.join(config.EVAL_DIR, dataset, filename_out), index=False, encoding='utf8')


def analyze_contrast_relations(dataset, filename):
    """Identifies the slots involved in a contrast relation."""

    contrast_connectors = ['but', 'however', 'yet']
    slots_before = []
    slots_after = []

    print('Analyzing contrast relations in ' + str(filename))

    # Read in the data
    data_cont = data_loader.init_test_data(os.path.join(config.DATA_DIR, dataset, filename))
    mrs_orig, utterances_orig = data_cont['data']
    _, _, slot_sep, val_sep, val_sep_end = data_cont['separators']

    # Preprocess the MRs
    mrs = [data_loader.convert_mr_from_str_to_list(mr, data_cont['separators']) for mr in mrs_orig]

    for mr, utt in zip(mrs, utterances_orig):
        mr_dict = OrderedDict()
        mr_list_augm = []

        # Extract the slot-value pairs into a dictionary
        for slot_value in mr.split(slot_sep):
            slot, value, slot_orig, value_orig = data_loader.parse_slot_and_value(slot_value, val_sep, val_sep_end)
            mr_dict[slot] = value
            mr_list_augm.append((slot, value_orig))

        # Find the slot alignment
        alignment = find_alignment(utt, mr_dict)

        slot_before = None
        slot_after = None

        for contrast_conn in contrast_connectors:
            contrast_pos = utt.find(contrast_conn)
            if contrast_pos >= 0:
                slot_before = None
                slot_after = None

                for pos, slot, value in alignment:
                    slot_before = slot_after
                    slot_after = slot

                    if pos > contrast_pos:
                        break

                break

        slots_before.append(slot_before if slot_before is not None else '')
        slots_after.append(slot_after if slot_after is not None else '')

    # Calculate the frequency distribution of slots involved in a contrast relation
    contrast_slot_cnt = Counter()
    contrast_slot_cnt.update(slots_before + slots_after)
    del contrast_slot_cnt['']
    print('\n---- Slot distribution in contrast relations ----\n')
    print('\n'.join(slot + ': ' + str(freq) for slot, freq in contrast_slot_cnt.most_common()))

    # Calculate the frequency distribution of slot pairs involved in a contrast relation
    contrast_slot_cnt = Counter()
    slot_pairs = [tuple(sorted(slot_pair)) for slot_pair in zip(slots_before, slots_after) if slot_pair != ('', '')]
    contrast_slot_cnt.update(slot_pairs)
    print('\n---- Slot pair distribution in contrast relations ----\n')
    print('\n'.join(slot_pair[0] + ', ' + slot_pair[1] + ': ' + str(freq) for slot_pair, freq in contrast_slot_cnt.most_common()))

    new_df = pd.DataFrame(columns=['mr', 'ref', 'slot before contrast', 'slot after contrast'])
    new_df['mr'] = mrs_orig
    new_df['ref'] = utterances_orig
    new_df['slot before contrast'] = slots_before
    new_df['slot after contrast'] = slots_after

    filename_out = os.path.splitext(filename)[0] + ' [contrast relations].csv'
    new_df.to_csv(os.path.join(config.DATA_DIR, dataset, filename_out), index=False, encoding='utf8')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dataset_path", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--dataset_partition", type=str, default=None, help="Name of the dataset partition (usually train, test or validation).")
    parser.add_argument("--dataset_name", type=str, help="Dataset name (either Viggo or E2E")
    parser.add_argument("--generated_file_path", type=str, help="Path to the file containing the generated sentences")
    parser.add_argument("--output_name", type=str, help="Output name")
    parser.add_argument("--output_dir_path", type=str, help="Output dir path")
    parser.add_argument("--datatuner_refs", action="store_true", help="Whether the refs are generated from the original Datatuner or not")
    parser.add_argument("--txt_predictions", action="store_true", help="Whether the refs are generated from raw txt")
    return parser.parse_args()


def extract_original_datatuner_refs(datatuner_refs_path: str) -> List[str]:
    refs = []
    with open(datatuner_refs_path, "r") as f:
        datatuner_json = json.load(f)
        for entry in datatuner_json:
            zipped_entry = list(zip(entry["pred"], entry["pred_prob"], entry["reranked"]))
            sorted_entry = sorted(zipped_entry, key=lambda x: (x[0], -x[1])) # pred, -pred_prob
            refs.append(sorted_entry[0][2])
    return refs


def extract_external_refs(external_refs_path: str) -> List[str]:
    generated_file_lines = []
    refs = []
    with open(external_refs_path, "r") as f:
        generated_file_lines = f.readlines()
    for line in generated_file_lines:
        if "GEN (default):" in line:
            refs.append(line.split("GEN (default):")[1].strip())
        elif "GEN:" in line:
            refs.append(line.split("GEN:")[1].strip())
    return refs


def extract_external_refs_from_json(external_refs_path: str) -> List[str]:
    predictions = []
    with open(external_refs_path, "r") as f:
        predictions = json.load(f) 
    refs = [prediction["gen"][0] for prediction in predictions]
    return refs


def merge_external_refs_with_mr(original_dataset_dir_path: str, dataset_filename: str, external_refs_path: str, 
                                txt_predictions: bool = False, datatuner_refs: bool = False) -> str:
    """Gets refs from an external file and then pairs them to the generated sentences
    in a pandas DataFrame object which is written to disk.
    The DataFrame is the same as the base dataset, but with the "ref" column 
    replaced with the generated sentences.

    Args:
        original_dataset_dir_path (str)
        dataset_filename (str)
        external_refs_path (str)
        txt_predictions (bool, optional): If the predictions/refs are taken from an (old version) txt file with generated sentences. Defaults to True.
        datatuner_refs (bool, optional): If the predictions/refs are taken from a DataTuner generated file. Defaults to False.

    Returns:
        str: path to the merged dataset
    """
    if not datatuner_refs:
        if txt_predictions:
            refs = extract_external_refs(external_refs_path)
        else:
            refs = extract_external_refs_from_json(external_refs_path)
    else:
        refs = extract_original_datatuner_refs(external_refs_path)
    df_data = pd.read_csv(os.path.join(original_dataset_dir_path, dataset_filename), header=0)
    df_data["ref"] = refs
    merged_file_path = os.path.join(original_dataset_dir_path, f"merged_{dataset_filename}")
    df_data.to_csv(merged_file_path, index=False, encoding='utf-8-sig')
    return merged_file_path


def main():
    args = parse_arguments()
    dataset_name = args.dataset_name.strip().lower()
    dataset_class = get_dataset_class(dataset_name)
    output_name = f"{dataset_name}_{args.output_name}"
    dataset_partition = args.base_dataset_path.split(os.sep)[-1].strip() if args.dataset_partition is None else args.dataset_partition
    output_name_without_file = f"{dataset_partition}_{output_name.split(os.sep)[0].strip()}"
    merged_dataset_path = merge_external_refs_with_mr(args.base_dataset_path, dataset_partition, args.generated_file_path, datatuner_refs=args.datatuner_refs, txt_predictions=args.txt_predictions)
    score_slot_realizations(args.base_dataset_path, merged_dataset_path, dataset_class, output_name=output_name_without_file, output_dir_path=args.output_dir_path, slot_level=True)


if __name__ == '__main__':
    main()
    # score_slot_realizations(r'/d/Git/data2text-nlg/data/multiwoz',
    #                         'valid.csv', MultiWOZDataset, slot_level=True)
    # score_slot_realizations(r'/d/Git/data2text-nlg/data/video_game',
    #                         'valid.csv', ViggoDataset, slot_level=True)
