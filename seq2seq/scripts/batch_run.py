import os

from seq2seq.data_loader import E2EDataset, E2ECleanedDataset, MultiWOZDataset, ViggoDataset
from seq2seq.scripts.slot_error_rate import calculate_slot_error_rate
from seq2seq.scripts.utterance_stats import utterance_stats
from seq2seq.slot_aligner.data_analysis import score_slot_realizations


def batch_calculate_slot_error_rate(input_dir, checkpoint_name, dataset_class, exact_matching=False, slot_level=False,
                                    verbose=False):
    files_processed = []
    ser_list = []

    decoding_suffixes = [
        '_no_beam_search',
        '_beam_search',
        # '_nucleus_sampling',
    ]
    was_reranking_used = True
    if 'gpt2' in os.path.split(input_dir)[-1]:
        length_penalty_vals = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    else:
        length_penalty_vals = [0.8, 0.9, 1.0, 1.5, 2.0, 3.0]
    p_vals = [0.3, 0.5, 0.8]

    for decoding_suffix in decoding_suffixes:
        reranking_suffixes = ['']
        if was_reranking_used and decoding_suffix != '_no_beam_search':
            reranking_suffixes.append('_reranked')

        for reranking_suffix in reranking_suffixes:
            if decoding_suffix == '_beam_search':
                value_suffixes = ['_' + str(val) for val in length_penalty_vals]
            elif decoding_suffix == '_nucleus_sampling':
                value_suffixes = ['_' + str(val) for val in p_vals]
            else:
                value_suffixes = ['']

            for value_suffix in value_suffixes:
                file_name = checkpoint_name + decoding_suffix + reranking_suffix + value_suffix + '.csv'
                files_processed.append(file_name)
                if verbose:
                    print(f'Running with file "{file_name}"...')

                if exact_matching:
                    ser = calculate_slot_error_rate(input_dir, file_name, dataset_class, slot_level=slot_level,
                                                    verbose=verbose)
                else:
                    ser = score_slot_realizations(input_dir, file_name, dataset_class, slot_level=slot_level,
                                                  verbose=verbose)
                ser_list.append(ser)

    if not verbose:
        # Print a summary of all files processed (in the same order)
        print()
        print('>> Files processed:')
        print('\n'.join(files_processed))


def batch_utterance_stats(input_dir, export_vocab=False, verbose=False):
    files_processed = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv') and '[errors]' not in file_name:
            files_processed.append(file_name)
            if verbose:
                print(f'Running with file "{file_name}"...')

            utterance_stats(os.path.join(input_dir, file_name), export_vocab=export_vocab, verbose=verbose)

            if verbose:
                print()

    if not verbose:
        # Print a summary of all files processed (in the same order)
        print()
        print('>> Files processed:')
        print('\n'.join(files_processed))


def run_batch_calculate_slot_error_rate():
    # input_dir = os.path.join('seq2seq', 'predictions', 'multiwoz', 'finetuned_verbalized_slots', 'bart-base_lr_1e-5_bs_32_wus_500_run4')
    # checkpoint_name = 'epoch_18_step_1749'
    # dataset_class = MultiWOZDataset

    input_dir = os.path.join('seq2seq', 'predictions', 'rest_e2e_cleaned', 'finetuned_verbalized_slots', 't5-small_lr_2e-4_bs_64_wus_100_run4')
    checkpoint_name = 'epoch_11_step_524'
    dataset_class = E2ECleanedDataset

    if 'multiwoz' in dataset_class.name:
        batch_calculate_slot_error_rate(
            input_dir, checkpoint_name, dataset_class, exact_matching=True, slot_level=False, verbose=False)
    else:
        batch_calculate_slot_error_rate(
            input_dir, checkpoint_name, dataset_class, exact_matching=False, slot_level=True, verbose=False)


def run_batch_utterance_stats():
    # input_dir = os.path.join('seq2seq', 'predictions_baselines', 'DataTuner', 'video_game')
    # input_dir = os.path.join('seq2seq', 'predictions', 'rest_e2e_cleaned', 'finetuned', 'gpt2_lr_2e-5_bs_20_wus_500_run1')
    input_dir = os.path.join('seq2seq', 'predictions', 'video_game', 'finetuned', 'gpt2_lr_2e-5_bs_16_wus_100_run4')

    batch_utterance_stats(input_dir, export_vocab=False, verbose=False)


if __name__ == '__main__':
    run_batch_calculate_slot_error_rate()
    # run_batch_utterance_stats()
