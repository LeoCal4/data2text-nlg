import argparse
import numpy as np
import os
import pandas as pd
# from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import yaml

from seq2seq.data_loader import E2EDataset, E2ECleanedDataset, ViggoDataset
from seq2seq.slot_aligner.slot_alignment import score_alignment
from seq2seq.task_config import TestConfig, TrainingConfig


def load_config(config_name):
    config_path = os.path.join('seq2seq', 'config', config_name + '.yaml')

    try:
        with open(config_path) as f_config:
            config = yaml.safe_load(f_config)
    except FileNotFoundError:
        print('Error: config file "{}" not found'.format(config_path))
        sys.exit()
    except yaml.YAMLError as err:
        print(err)

    return config


def load_pretrained_gpt2_model_and_tokenizer(model_name, special_tokens=None):
    # Load pretrained tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    special_tokens = {
        'bos_token': '<|begoftext|>',
        'pad_token': '<PAD>',
        'additional_special_tokens': special_tokens
    }
    tokenizer.add_special_tokens(special_tokens)

    # Load pretrained model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_model_checkpoint(model, model_name, epoch, step):
    model_dir = os.path.join('seq2seq', 'model')
    if not os.path.exists(model_dir):
        raise NotADirectoryError('No saved checkpoint found')

    file_name = '{}_epoch_{}_step_{}.pt'.format(model_name, epoch, step)
    checkpoint_path = os.path.join(model_dir, file_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('Checkpoint "{}" not found'.format(file_name))

    model.load_state_dict(torch.load(checkpoint_path))


def save_model(model, model_name, epoch, step):
    model_dir = os.path.join('seq2seq', 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_name = '{}_epoch_{}_step_{}.pt'.format(model_name, epoch, step)
    torch.save(model.state_dict(), os.path.join(model_dir, file_name))


def create_label_mask(input_ids, input_mask, label_mask):
    label_offsets = input_mask.sum(dim=1) - label_mask.sum(dim=1)

    # DEBUG
    # print('>> label offsets:', label_offsets)

    mask = torch.zeros_like(input_ids)
    mask[torch.arange(input_ids.shape[0]), label_offsets] = 1
    mask = 1 - mask.cumsum(dim=1)

    # DEBUG
    # print('>> label mask:', mask)

    return mask


def train(config, dataset_class, device='cpu'):
    global_step = 0

    # Load model and corresponding tokenizer
    model, tokenizer = load_pretrained_gpt2_model_and_tokenizer(
        config.pretrained_model,
        special_tokens=dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names))
    model = model.to(device)

    # Load training and validation data
    train_set = dataset_class(tokenizer, 'train', lowercase=True, convert_slot_names=config.convert_slot_names)
    train_data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)

    valid_set = dataset_class(tokenizer, 'valid', lowercase=True, convert_slot_names=config.convert_slot_names)
    valid_data_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Set up the optimizer and learning rate scheduler
    num_training_steps = len(train_data_loader) * config.num_epochs
    optimizer = AdamW(model.parameters(), lr=config.lr, eps=1e-6, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=config.num_warmup_steps,
                                                num_training_steps=num_training_steps)

    if config.fp16:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, config.num_epochs + 1):
        print()
        print(' *************** ')
        print('**   EPOCH {:<2}  **'.format(epoch))
        print(' *************** ')
        print()

        train_loss_sum = 0

        for step, batch in enumerate(tqdm(train_data_loader, desc='Step')):
            # tokenizer.padding_side = 'left'

            inputs = tokenizer(batch[0], add_special_tokens=False, padding=True, truncation=True,
                               max_length=config.max_seq_length, return_tensors='pt')
            mrs_only = tokenizer(batch[1], add_special_tokens=False, padding=True, truncation=True,
                                 max_length=config.max_seq_length, return_tensors='pt')

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            mr_mask = mrs_only['attention_mask']

            label_mask = torch.zeros_like(input_ids)
            label_mask[:, :mr_mask.shape[1]] = mr_mask
            label_mask[input_ids == tokenizer.pad_token_id] = 1

            label_ids = input_ids.masked_fill(label_mask, -100)
            # label_ids = input_ids.clone()

            input_tensor = input_ids.to(device)
            mask_tensor = input_mask.to(device)
            label_tensor = label_ids.to(device)

            # DEBUG
            # print('>> ENCODED inputs:', input_ids)
            # print('>> ENCODED labels:', label_ids)
            # print('>> DECODED inputs:', tokenizer.decode(input_ids[0]))
            # print('>> DECODED labels:', tokenizer.decode(label_ids[0]))

            model.train()

            # Clear previously calculated gradients (must perform before a backward pass, unless using RNNs)
            model.zero_grad()

            if config.fp16:
                # Forward pass
                with torch.cuda.amp.autocast():
                    loss = model(input_tensor, attention_mask=mask_tensor, labels=label_tensor)[0]

                # Accumulate the training loss
                train_loss_sum += loss.item()

                # Backward pass
                scaler.scale(loss).backward()

                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)

                # Clip the norm of the gradients (in order to prevent the gradients from exploding)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                loss = model(input_tensor, attention_mask=mask_tensor, labels=label_tensor)[0]

                # Accumulate the training loss
                train_loss_sum += loss.item()

                # Backward pass
                loss.backward()

                # Clip the norm of the gradients (in order to prevent the gradients from exploding)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()

            # Update the learning rate according to the defined schedule
            scheduler.step()

            global_step += 1

            if global_step % config.eval_interval_in_steps == 0:
                # Print stats
                avg_train_loss = train_loss_sum / config.eval_interval_in_steps
                print()
                print('>> Training loss:  \t{0:.4f}'.format(avg_train_loss))
                print()
                train_loss_sum = 0

                # Validation
                metrics = evaluate(valid_set, valid_data_loader, model, tokenizer, device=device)
                print()
                print('>> Validation loss: {0:.4f}'.format(metrics.get('loss').item()))
                print('>> Validation PPL: {0:.4f}'.format(metrics.get('perplexity').item()))
                print('>> Validation BLEU: {0:.4f}'.format(metrics.get('bleu')))
                print('>> Validation BLEU (multi-ref): {0:.4f}'.format(metrics.get('bleu_multiref')))
                print()

                save_model(model, config.pretrained_model, epoch, step + 1)

        save_model(model, config.pretrained_model, epoch, len(train_data_loader))

    model_dir = os.path.join('seq2seq', 'model', 'final')
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def evaluate(dataset, data_loader, model, tokenizer, device='cpu'):
    eval_loss_sum = 0.0
    num_steps = 0
    predictions = []
    model.eval()

    for batch in tqdm(data_loader, desc='Evaluating'):
        inputs = tokenizer(batch[0], add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')
        mrs_only = tokenizer(batch[1], add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')

        input_ids = inputs['input_ids']
        input_mask = inputs['attention_mask']
        mr_mask = mrs_only['attention_mask']

        label_mask = torch.zeros_like(input_ids)
        label_mask[:, :mr_mask.shape[1]] = mr_mask
        label_mask[input_ids == tokenizer.pad_token_id] = 1

        label_ids = input_ids.masked_fill(label_mask, -100)
        # label_ids = input_ids.clone()

        input_tensor = input_ids.to(device)
        mask_tensor = input_mask.to(device)
        label_tensor = label_ids.to(device)

        # input_ids, attention_mask, label_ids, label_mask = batch

        with torch.no_grad():
            model_outputs = model(input_tensor,
                                  attention_mask=mask_tensor,
                                  labels=label_tensor)
            loss, logits = model_outputs[:2]

            # Accumulate the evaluation loss
            eval_loss_sum += loss.item()

            logits = logits.detach().cpu().numpy()

            # DEBUG
            # print('>> logits.shape:', logits.shape)
            # print('>> logits:', logits)
            # print('>> type(label_mask):', type(label_mask))

            # Shift label mask one position to the left, ignoring thus the last position of the logits
            logits = logits[:, :-1, :]
            logit_masks = label_mask.numpy()[:, 1:]

            for logit_array, logit_mask in zip(logits, logit_masks):
                # print('>> SHAPE logits (before mask):', logit_array.shape)
                # print('>> SHAPE mask:', logit_mask.shape)
                # print('>> MASK:', mask_array)
                logit_array = logit_array[logit_mask == 0]
                # print('>> SHAPE logits (after mask):', logit_array.shape)
                # print('>> LOGITS (after mask):', logit_array)
                output_ids = np.argmax(logit_array, axis=1)
                # print('>> LOGITS (after argmax):', logit_array)
                prediction = tokenizer.decode(output_ids, skip_special_tokens=True)
                # print('>> PREDICTION:', prediction)
                predictions.append(prediction)

        num_steps += 1

    # DEBUG
    # print('>> PREDICTIONS:\n', predictions[:20])

    eval_loss = torch.tensor(eval_loss_sum / num_steps)
    perplexity = torch.exp(eval_loss)
    bleu = corpus_bleu(predictions, [dataset.get_utterances(lowercased=True)]).score
    bleu_multiref = calculate_multiref_bleu(dataset, predictions)

    result = {
        'loss': eval_loss,
        'perplexity': perplexity,
        'bleu': bleu,
        'bleu_multiref': bleu_multiref
    }

    return result


def calculate_multiref_bleu(dataset, predictions):
    # Group references and generated utterances by MR -- in order to perform multi-reference BLEU evaluation
    df_data = pd.DataFrame(zip(dataset.get_mrs(lowercased=True), dataset.get_utterances(lowercased=True), predictions),
                           columns=['mr', 'ref', 'out'])
    df_grouped_by_mr = df_data.groupby('mr', sort=False).agg(
        {'ref': (lambda x: list(x)), 'out': (lambda x: list(x))}).reset_index()

    references = df_grouped_by_mr['ref'].tolist()
    utterances = df_grouped_by_mr['out'].tolist()

    # Only works if the number of references is the same for each input
    # references_transposed = list(map(list, zip(*references)))

    max_num_refs = max(len(ref_list) for ref_list in references)
    references_transposed = [[] for _ in range(max_num_refs)]
    for ref_list in references:
        idx = 0
        for ref in ref_list:
            references_transposed[idx].append(ref)
            idx += 1

        # Pad with the first reference
        for i in range(idx, max_num_refs):
            references_transposed[i].append(ref_list[0])

    utterances_first = [utt[0] for utt in utterances]

    return corpus_bleu(utterances_first, references_transposed).score


def test(config, dataset_class, device='cpu'):
    predictions = []

    # Load model and corresponding tokenizer
    model, tokenizer = load_pretrained_gpt2_model_and_tokenizer(
        config.pretrained_model,
        special_tokens=dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names))
    load_model_checkpoint(model, config.pretrained_model, config.checkpoint_epoch, config.checkpoint_step)
    model = model.to(device)
    model.eval()

    # Load test data
    test_set = dataset_class(tokenizer, 'test', lowercase=True, convert_slot_names=config.convert_slot_names,
                             group_by_mr=True)
    test_data_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    for batch in tqdm(test_data_loader, desc='Evaluating'):
        inputs = tokenizer(batch[0], add_special_tokens=False, padding=True, truncation=True, return_tensors='pt')

        # DEBUG
        # print('>> ENCODED inputs:', inputs['input_ids'])
        # print('>> ENCODED mask:', inputs['attention_mask'])

        input_tensor = inputs['input_ids'].to(device)
        mask_tensor = inputs['attention_mask'].to(device)

        outputs = model.generate(input_tensor,
                                 attention_mask=mask_tensor,
                                 max_length=config.max_seq_length,
                                 num_beams=config.num_beams,
                                 early_stopping=config.early_stopping,
                                 no_repeat_ngram_size=config.no_repeat_ngram_size,
                                 do_sample=config.do_sample,
                                 top_p=config.top_p,
                                 top_k=config.top_k,
                                 temperature=config.temperature,
                                 repetition_penalty=config.repetition_penalty,
                                 length_penalty=config.length_penalty,
                                 num_return_sequences=config.num_return_sequences,
                                 bos_token_id=tokenizer.bos_token_id,
                                 pad_token_id=tokenizer.pad_token_id)

        # DEBUG
        # print('>> OUTPUTS', outputs)

        outputs_decoded = []

        # TODO: decode all outputs at the same time (for when num_return_sequences > 1)
        # TODO: generalize to batch_size > 1
        for i, output_seq in enumerate(outputs):
            utt_beg_pos = np.where(output_seq.cpu().numpy() == tokenizer.bos_token_id)[0][0] + 1
            utt_decoded = tokenizer.decode(output_seq[utt_beg_pos:], skip_special_tokens=True)
            outputs_decoded.append(utt_decoded)
            # print('>> Sample #{}: {}'.format(i, utt_decoded))

        predictions.append(outputs_decoded)

    # Make sure the output directory exists for the given dataset
    predictions_dir = os.path.join('seq2seq', 'predictions', test_set.name)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    # Prepare the metrics script command, and create the reference file for the given dataset
    eval_dir = os.path.join('seq2seq', 'eval')
    metrics_script = 'python ' + os.path.join(eval_dir, 'E2E', 'measure_scores.py')
    reference_file = os.path.join(eval_dir, 'test_references_{}.txt'.format(test_set.name))
    if not os.path.exists(reference_file):
        print('>> Generating a reference file for the "{}" test set.'.format(test_set.name))
        test_set.create_reference_file_for_testing()

    eval_configurations = []

    if config.semantic_reranking:
        predictions_reranked = rerank_beams(predictions, test_set.get_mrs_as_dicts())
        predictions_reranked = [pred_beam[0] for pred_beam in predictions_reranked]
        eval_configurations.append((predictions_reranked, True))

    # For the evaluation of non-reranked predictions select the top candidate from the generated pool
    predictions = [pred_beam[0] for pred_beam in predictions]
    eval_configurations.insert(0, (predictions, False))

    for prediction_list, reranked in eval_configurations:
        file_name_root = compose_output_file_name(config, reranked=reranked)

        # Save generated utterances along with their corresponding MRs into a CSV file
        file_name = f'{file_name_root}.csv'
        df_predictions = pd.DataFrame({'mr': test_set.get_mrs(raw=True), 'utt': prediction_list})
        df_predictions.to_csv(os.path.join(predictions_dir, file_name), index=False, encoding='utf-8-sig')

        # Save generated utterances in a text file (for reference-based metric evaluation)
        file_name = f'{file_name_root}_utt_only.txt'
        predictions_file = os.path.join(predictions_dir, file_name)
        with open(predictions_file, 'w') as f_out:
            for prediction in prediction_list:
                f_out.write(prediction + '\n')

        # Run the metrics script provided by the E2E NLG Challenge
        os.system(metrics_script + ' ' + reference_file + ' ' + predictions_file)


def compose_output_file_name(config, reranked=False):
    if config.num_beams > 1:
        inference_method_suffix = '_beam_search_'
        if reranked:
            inference_method_suffix += 'reranked_'
        inference_method_suffix += str(config.length_penalty)
    elif config.do_sample and config.top_p < 1.0:
        inference_method_suffix = '_nucleus_sampling_'
        if reranked:
            inference_method_suffix += 'reranked_'
        inference_method_suffix += str(config.top_p)
    elif config.do_sample and config.top_k > 0:
        inference_method_suffix = '_top_k_sampling_'
        if reranked:
            inference_method_suffix += 'reranked_'
        inference_method_suffix += str(config.top_k)
    else:
        inference_method_suffix = '_no_beam_search'

    file_name = 'epoch_{}_step_{}{}'.format(config.checkpoint_epoch, config.checkpoint_step, inference_method_suffix)

    return file_name


def generate_from_input(input_str, config, dataset_class, device='cpu'):
    # Load model and corresponding tokenizer
    model, tokenizer = load_pretrained_gpt2_model_and_tokenizer(
        config.pretrained_model,
        special_tokens=dataset_class.get_special_tokens(convert_slot_names=config.convert_slot_names))
    load_model_checkpoint(model, config.pretrained_model, config.checkpoint_epoch, config.checkpoint_step)
    model = model.to(device)
    model.eval()

    input_ids = tokenizer(input_str)['input_ids']
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

    outputs = model.generate(input_tensor,
                             max_length=config.max_seq_length,
                             num_beams=config.num_beams,
                             early_stopping=config.early_stopping,
                             no_repeat_ngram_size=config.no_repeat_ngram_size,
                             do_sample=config.do_sample,
                             top_p=config.top_p,
                             top_k=config.top_k,
                             temperature=config.temperature,
                             repetition_penalty=config.repetition_penalty,
                             length_penalty=config.length_penalty,
                             num_return_sequences=config.num_return_sequences,
                             bos_token_id=tokenizer.bos_token_id,
                             pad_token_id=tokenizer.pad_token_id)

    for i, output_seq in enumerate(outputs):
        utt_beg_pos = np.where(output_seq.cpu().numpy() == tokenizer.bos_token_id)[0][0] + 1
        utt_decoded = tokenizer.decode(output_seq[utt_beg_pos:], skip_special_tokens=True)
        print('>> Sample #{}: {}'.format(i, utt_decoded))


def rerank_beams(beams, mrs, keep_n=None, keep_least_errors_only=False):
    """Reranks beams based on the slot error rate determined by the slot aligner. Keeps at most n best candidates."""
    beams_reranked = []

    for idx, mr in enumerate(tqdm(mrs, desc='Reranking')):
        beam_scored = []

        for utt in beams[idx]:
            # Calculate the slot error score
            score = score_alignment(utt, mr)
            beam_scored.append((utt, score))

        # Rerank utterances by slot error score
        beam_scored.sort(key=lambda tup: tup[1], reverse=True)

        if keep_least_errors_only:
            # Filter only those utterances that have the least number of errors identified by the slot aligner
            beam_scored = [candidate for candidate in beam_scored if candidate[1] == beam_scored[0][1]]

        # Keep at most n candidates
        if keep_n is not None and len(beam_scored) > keep_n > 0:
            beam_scored = beam_scored[:keep_n]

        # DEBUG
        # if idx < 5:
        #     print('>> Scored beams:')
        #     print('\n'.join('{0} :: {1}'.format(utt[1], utt[0]) for utt in beam_scored))
        #     print()

        # Store the reranked beam (utterances only)
        beams_reranked.append([utt[0] for utt in beam_scored])

    return beams_reranked


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Training/test config name')
    parser.add_argument('-d', '--dataset', required=True, choices=['rest_e2e', 'rest_e2e_cleaned', 'video_game'],
                        help='Dataset name')
    parser.add_argument('-t', '--task', required=True, choices=['train', 'test'],
                        help='Task (train or test)')
    args = parser.parse_args()

    # Load the task configuration
    config = load_config(args.config)

    # Get the corresponding dataset class
    if args.dataset == 'rest_e2e':
        dataset_class = E2EDataset
    elif args.dataset == 'rest_e2e_cleaned':
        dataset_class = E2ECleanedDataset
    elif args.dataset == 'video_game':
        dataset_class = ViggoDataset
    else:
        print('Error: dataset "{}" not recognized'.format(args.dataset))
        sys.exit()

    # Set the device to GPU if available, or CPU otherwise
    if torch.cuda.is_available():
        device = 'cuda'
        print('GPUs available:', torch.cuda.device_count())
        print('CUDA version:', torch.version.cuda)
    else:
        device = 'cpu'

    # Run the corresponding task
    if args.task == 'train':
        train(TrainingConfig(config), dataset_class, device=device)
    elif args.task == 'test':
        test(TestConfig(config), dataset_class, device=device)
    elif args.task == 'generate':
        input_str = '<|name|> alimentum <|area|> city centre <|familyfriendly|> no <|begoftext|>'
        generate_from_input(input_str, TestConfig(config), dataset_class, device=device)
    else:
        print('Error: task "{}" not recognized'.format(args.task))
        sys.exit()


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    main()