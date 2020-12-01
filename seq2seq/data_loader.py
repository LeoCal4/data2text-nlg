from collections import defaultdict
import os
import pandas as pd
import re
import regex
from torch.utils.data import Dataset


COMMA_PLACEHOLDER = ' __comma__'


class MRToTextDataset(Dataset):
    """Seq-to-seq dataset with flat structured meaning representation (MR) as input and natural text as output."""
    name = 'mr_to_text'
    delimiters = {}

    def __init__(self, tokenizer, partition='train', lowercase=False, convert_slot_names=False, group_by_mr=False,
                 no_target=False, separate_source_and_target=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.partition = partition

        self.convert_to_lowercase = lowercase
        self.convert_slot_names = convert_slot_names
        self.group_by_mr = group_by_mr
        self.no_target = no_target
        self.separate_source_and_target = separate_source_and_target

        self.mrs_raw = []
        self.mrs_raw_as_lists = []
        self.mrs = []
        self.mrs_as_lists = []
        self.utterances = []

        self.load_data()

    def __len__(self):
        return len(self.mrs)

    def __getitem__(self, idx):
        mr = self.mrs[idx]
        utt = self.utterances[idx] if self.utterances else None

        if self.separate_source_and_target:
            source_str = mr
            if self.no_target:
                target_str = ''
            elif utt is not None:
                target_str = utt
            else:
                raise ValueError('Utterances must be present in training and validation data')
        else:
            if self.no_target:
                # If test set, load the MRs only as source
                source_str = mr + self.bos_token
            elif utt is not None:
                # If training/validation set, concatenate the MR and the utterance with a BOS token in between
                source_str = mr + self.bos_token + utt + self.eos_token
            else:
                raise ValueError('Utterances must be present in training and validation data')
            # When MR and utterance are concatenated as source, use the target string as an auxiliary variable
            target_str = mr + self.bos_token

        return source_str, target_str

    def load_data(self):
        # Load the data file
        dataset_path = self.get_data_file_path(self.partition)
        df_data = pd.read_csv(dataset_path, header=0, encoding='utf8')

        # Read the MRs and utterances from the file
        self.mrs_raw, self.utterances = self.read_data_from_dataframe(df_data, group_by_mr=self.group_by_mr)

        # Convert MRs to an intermediate format of lists of tuples
        self.mrs_raw_as_lists = [self.parse_mr_to_list_of_tuples(mr) for mr in self.mrs_raw]

        # Perform dataset-specific preprocessing of the MRs, and convert them back to strings
        self.mrs = self.get_mrs(lowercase=self.convert_to_lowercase, convert_slot_names=self.convert_slot_names)

        # Lowercase utterances if needed
        self.utterances = self.get_utterances(lowercase=self.convert_to_lowercase)

        # DEBUG
        # print('>> MRs:\n{}'.format('\n'.join(self.mrs[:50])))
        # if isinstance(self.utterances[0], str):
        #     print('>> Utterances:\n{}'.format('\n'.join(self.utterances[:10])))
        # else:
        #     print('>> Utterances:\n{}'.format('\n'.join(['[' + '; '.join(utt) + ']' for utt in self.utterances[:10]])))

        if self.utterances:
            assert len(self.mrs) == len(self.utterances)

        # DEBUG
        # self.mrs = self.mrs[:10]
        # self.mrs_raw = self.mrs_raw[:10]
        # self.mrs_as_lists = self.mrs_as_lists[:10]
        # self.utterances = self.utterances[:10]

    @staticmethod
    def get_data_file_path(partition):
        raise NotImplementedError('method \'get_data_file_path\' must be defined by subclass')

    @staticmethod
    def read_data_from_dataframe(df_data, group_by_mr=False):
        # Extract the column names
        mr_col_name = df_data.columns[0]
        utt_col_name = df_data.columns[1] if df_data.shape[1] > 1 else None

        # Save the MRs and the utterances as lists (repeated MRs are collapsed for test data)
        if group_by_mr:
            # If utterances are present in the data
            if df_data.shape[1] > 1:
                # Group by MR, and aggregate utterances into lists
                df_grouped_by_mr = df_data.groupby(mr_col_name, sort=False)[utt_col_name].apply(list).reset_index()
                mrs = df_grouped_by_mr[mr_col_name].tolist()
                utterances = df_grouped_by_mr[utt_col_name].tolist()
            else:
                mrs = df_data[mr_col_name].tolist()
                utterances = []
        else:
            mrs = df_data[mr_col_name].tolist()
            if df_data.shape[1] > 1:
                utterances = df_data[utt_col_name].tolist()
            else:
                raise ValueError('Training and validation input data are expected to have two columns')

        return mrs, utterances

    def parse_mr_to_list_of_tuples(self, mr_as_str):
        mr_as_list = []

        mr_as_str = self.preprocess_da_in_mr(mr_as_str)

        # Replace commas in values if comma is the slot separator
        if self.delimiters['slot_sep'].strip() == ',' and self.delimiters.get('val_end') is not None:
            mr_as_str = self.replace_commas_in_slot_values(
                mr_as_str, self.delimiters['val_beg'], self.delimiters['val_end'])

        # Extract the sequence of slots and their corresponding values
        for slot_value_pair in mr_as_str.split(self.delimiters['slot_sep']):
            slot, value = self.parse_slot_and_value(slot_value_pair)
            mr_as_list.append((slot, value))

        return mr_as_list

    def preprocess_slot_names_in_mr(self, mr_as_list, convert_slot_names=False):
        mr_processed = []

        for slot, value in mr_as_list:
            if convert_slot_names:
                slot = self.convert_slot_name_to_special_token(slot)
            else:
                if slot == 'da':
                    slot = 'intent'
                    value = self.verbalize_da_name(value)
                else:
                    slot = self.verbalize_slot_name(slot)

            # if slot in mr_dict:
            #     slot_new = slot
            #     slot_ctr = 1
            #     while slot_new in mr_dict:
            #         slot_new = slot + str(slot_ctr)
            #         slot_ctr += 1
            #     slot = slot_new

            mr_processed.append((slot, value))

        return mr_processed

    @staticmethod
    def convert_mr_dict_to_str(mr_dict):
        return ' '.join(['{0}{1}'.format(slot, ' ' + val if val else '') for slot, val in mr_dict.items()])

    @staticmethod
    def convert_mr_list_to_str(mr_list, add_separators=False):
        slot_sep = ' | ' if add_separators else ' '
        val_sep = ' = ' if add_separators else ' '

        return slot_sep.join(['{0}{1}'.format(slot, val_sep + val if val else '') for slot, val in mr_list])

    @classmethod
    def preprocess_da_in_mr(cls, mr):
        """Converts the DA type indication(s) in the MR to the slot-and-value format."""

        # If no DA type indication is expected in the data, return the MR unchanged
        if cls.delimiters.get('da_beg') is None:
            return mr

        mr_new = ''

        if cls.delimiters.get('da_sep') is None:
            # Parse a single DA with its slots and values
            match = regex.match(r'(\S+){0}(.*?){1}$'.format(
                re.escape(cls.delimiters['da_beg']), re.escape(cls.delimiters['da_end'])), mr)
        else:
            # Parse multiple DAs with their respective slots and values
            match = regex.match(r'(\S+){0}(.*?){1}(?:{2}(\S+){0}(.*?){1})*'.format(
                re.escape(cls.delimiters['da_beg']), re.escape(cls.delimiters['da_end']),
                re.escape(cls.delimiters['da_sep'])), mr)

        if not match:
            print(f'Warning: Unexpected format of the following MR:\n{mr}')
            return mr

        for i in range(1, len(match.groups()) + 1, 2):
            if match.group(i) is None:
                break

            for j in range(len(match.captures(i))):
                da_type = match.captures(i)[j]
                slot_value_pairs = match.captures(i + 1)[j]

                if i > 1:
                    mr_new += cls.delimiters['da_sep']

                # Convert the extracted DA type to the slot-value form and prepend it to the DA's slots and values
                mr_new += 'da' + cls.delimiters['val_beg'] + da_type
                if cls.delimiters.get('val_end') is not None:
                    mr_new += cls.delimiters['val_end']
                if len(slot_value_pairs) > 0:
                    mr_new += cls.delimiters['slot_sep'] + slot_value_pairs

        return mr_new

    @classmethod
    def parse_slot_and_value(cls, slot_value_pair_str):
        """Parses out the slot name and the slot value from a string representing this pair."""
        delim_idx = slot_value_pair_str.find(cls.delimiters['val_beg'])
        if delim_idx > -1:
            # Parse the slot
            slot = slot_value_pair_str[:delim_idx].strip()
            # Parse the value
            if cls.delimiters.get('val_end') is not None:
                value = slot_value_pair_str[delim_idx + 1:-1].strip()
            else:
                value = slot_value_pair_str[delim_idx + 1:].strip()
        else:
            # Parse the slot
            if cls.delimiters.get('val_end') is not None:
                slot = slot_value_pair_str[:-1].strip()
            else:
                slot = slot_value_pair_str.strip()
            # Set the value to the empty string
            value = ''

        value = value.replace(COMMA_PLACEHOLDER, ',')

        return slot, value

    @staticmethod
    def replace_commas_in_slot_values(mr, val_sep, val_sep_end):
        mr_new = ''
        val_beg_cnt = 0
        val_end_cnt = 0

        for c in mr:
            # If comma inside a value, replace the comma with placeholder
            if c == ',' and val_beg_cnt > val_end_cnt:
                mr_new += COMMA_PLACEHOLDER
                continue

            # Keep track of value beginning and end
            if c == val_sep:
                val_beg_cnt += 1
            elif c == val_sep_end:
                val_end_cnt += 1

            mr_new += c

        return mr_new

    @staticmethod
    def put_back_commas_in_mr_values(mrs):
        return [mr.replace(COMMA_PLACEHOLDER, ',') for mr in mrs]

    @staticmethod
    def lowercase_mrs(mrs):
        """Lowercases the given MRs."""
        if isinstance(mrs[0], list):
            return [[(slot.lower(), value.lower()) for slot, value in mr_as_list] for mr_as_list in mrs]
        elif isinstance(mrs[0], str):
            return [mr_as_str.lower() for mr_as_str in mrs]
        else:
            raise TypeError('MRs must be strings, or lists of slot-and-value tuples.')

    @staticmethod
    def lowercase_utterances(utterances):
        """Lowercases the given utterances."""
        if isinstance(utterances[0], str):
            return [utt.lower() for utt in utterances]
        elif isinstance(utterances[0], list):
            return [[utt.lower() for utt in utt_list] for utt_list in utterances]
        else:
            raise TypeError('Utterances must be strings, or lists of strings.')

    def create_reference_file_for_testing(self):
        """Creates a text file with groups of utterances corresponding to one MR separated by an empty line."""
        eval_dir = os.path.join('seq2seq', 'eval')
        out_file = os.path.join(eval_dir, 'test_references_{}.txt'.format(self.name))

        with open(out_file, 'w', encoding='utf8') as f_out:
            if isinstance(self.utterances[0], list):
                for i in range(len(self.utterances)):
                    f_out.write('\n'.join(self.utterances[i]))
                    f_out.write('\n\n')
            elif isinstance(self.utterances[0], str):
                f_out.write('\n\n'.join(self.utterances))
                f_out.write('\n')

    def get_mrs(self, raw=False, as_lists=False, lowercase=False, convert_slot_names=False):
        if raw:
            mrs = self.mrs_raw
            if as_lists or lowercase:
                print('Warning: raw MRs are returned as strings with original letter case.')
        else:
            mrs = [self.preprocess_slot_names_in_mr(mr, convert_slot_names=convert_slot_names)
                   for mr in self.mrs_raw_as_lists]
            if lowercase:
                mrs = self.lowercase_mrs(mrs)
            if not as_lists:
                mrs = [self.convert_mr_list_to_str(mr, add_separators=(not convert_slot_names)) for mr in mrs]

        return mrs

    def get_utterances(self, lowercase=False):
        if lowercase:
            return self.lowercase_utterances(self.utterances)
        else:
            return self.utterances

    @classmethod
    def get_ontology(cls):
        ontology = defaultdict(set)

        train_set_path = cls.get_data_file_path('train')
        df_data = pd.read_csv(train_set_path, header=0, encoding='utf8')

        for mr_str in df_data[df_data.columns[0]]:
            mr_str = cls.preprocess_da_in_mr(mr_str)

            # Replace commas in values if comma is the slot separator
            if cls.delimiters['slot_sep'].strip() == ',' and cls.delimiters.get('val_end') is not None:
                mr_str = cls.replace_commas_in_slot_values(mr_str, cls.delimiters['val_beg'],
                                                           cls.delimiters['val_end'])

            for slot_value_pair in mr_str.split(cls.delimiters['slot_sep']):
                slot, value = cls.parse_slot_and_value(slot_value_pair)
                ontology[slot].add(value)

        return ontology

    @classmethod
    def get_special_tokens(cls, convert_slot_names=False):
        slot_tokens = set()

        if convert_slot_names:
            train_set_path = cls.get_data_file_path('train')
            df_data = pd.read_csv(train_set_path, header=0, encoding='utf8')

            for mr_str in df_data[df_data.columns[0]]:
                mr_str = cls.preprocess_da_in_mr(mr_str)

                # Replace commas in values if comma is the slot separator
                if cls.delimiters['slot_sep'].strip() == ',' and cls.delimiters.get('val_end') is not None:
                    mr_str = cls.replace_commas_in_slot_values(mr_str, cls.delimiters['val_beg'], cls.delimiters['val_end'])

                for slot_value_pair in mr_str.split(cls.delimiters['slot_sep']):
                    slot, _ = cls.parse_slot_and_value(slot_value_pair)
                    slot = cls.convert_slot_name_to_special_token(slot)
                    slot_tokens.add(slot)

        # DEBUG
        # print('>> slot_tokens:', slot_tokens)

        return sorted(list(slot_tokens))

    @staticmethod
    def verbalize_da_name(da_name):
        raise NotImplementedError('method \'verbalize_da_name\' must be defined by subclass')

    @staticmethod
    def verbalize_slot_name(slot_name):
        raise NotImplementedError('method \'verbalize_slot_name\' must be defined by subclass')

    @staticmethod
    def convert_slot_name_to_special_token(slot_name):
        """Converts a slot name to a special token."""
        return '<|{}|>'.format(slot_name.replace(' ', '').lower())


class E2EDataset(MRToTextDataset):
    """An MR-to-text dataset in the restaurant domain (provided as part of the E2E NLG Challenge)."""
    name = 'rest_e2e'
    delimiters = {
        'da_beg': None,
        'da_end': None,
        'da_sep': None,
        'slot_sep': ', ',
        'val_beg': '[',
        'val_end': ']'
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'rest_e2e')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'devset.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'testset.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'trainset.csv')

        return dataset_path

    @staticmethod
    def verbalize_slot_name(slot_name):
        slots_to_override = {
            'eatType': 'eatery type',
            'familyFriendly': 'family-friendly',
            'priceRange': 'price range',
        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name

        return slot_name_verbalized


class E2ECleanedDataset(E2EDataset):
    """A cleaned version of the E2E dataset in the restaurant domain."""
    name = 'rest_e2e_cleaned'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'rest_e2e_cleaned')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'devel-fixed.no-ol.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test-fixed.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train-fixed.no-ol.csv')

        return dataset_path


class MultiWOZDataset(MRToTextDataset):
    """A multi-domain dataset of task-oriented dialogues."""
    name = 'multiwoz'
    delimiters = {
        'da_beg': '(',
        'da_end': ')',
        'da_sep': ', ',
        'slot_sep': ', ',
        'val_beg': '[',
        'val_end': ']'
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'multiwoz')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'valid.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train.csv')

        return dataset_path


class ViggoDataset(MRToTextDataset):
    """An MR-to-text dataset in the video game domain."""
    name = 'video_game'
    delimiters = {
        'da_beg': '(',
        'da_end': ')',
        'da_sep': None,
        'slot_sep': ', ',
        'val_beg': '[',
        'val_end': ']'
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'video_game')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'valid.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train.csv')

        return dataset_path

    @staticmethod
    def verbalize_da_name(da_name):
        return da_name.replace('_', ' ')

    @staticmethod
    def verbalize_slot_name(slot_name):
        slots_to_override = {
            'esrb': 'ESRB rating',
            'exp_release_date': 'expected release date',
        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name.replace('_', ' ')
            for tok in ['linux', 'mac', 'steam']:
                slot_name_verbalized = re.sub(r'\b{}\b'.format(re.escape(tok)), tok.capitalize(), slot_name_verbalized)

        return slot_name_verbalized


class ViggoWithE2EDataset(ViggoDataset):
    """The ViGGO dataset with the training set merged with that of the E2E dataset."""
    name = 'video_game'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_with_e2e.csv')
        else:
            dataset_path = super(ViggoWithE2EDataset, ViggoWithE2EDataset).get_data_file_path(partition)

        return dataset_path


class Viggo20Dataset(ViggoDataset):
    """A 20% sample of the ViGGO dataset."""
    name = 'video_game_20'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('seq2seq', 'data', 'video_game')
        if partition == 'train':
            dataset_path = os.path.join(dataset_dir, 'train_sampled_0.2.csv')
        else:
            dataset_path = super(Viggo20Dataset, Viggo20Dataset).get_data_file_path(partition)

        return dataset_path
