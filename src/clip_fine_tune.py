import os
'''
Manually limiting the thread number for numpy
this is recommended if your CPU has many threads
'''
num_numpy_threads = '8'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads

from comet_ml import Experiment
import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, geometric_mean, harmonic_mean
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import base_path, squarepad_transform, targetpad_transform, CIRRDataset, FashionIQDataset
from utils import collate_fn, update_train_running_results, set_train_bar_description, extract_index_features, \
    save_model, generate_randomized_fiq_caption, element_wise_sum, device, cosine_lr_schedule
from validate import compute_cirr_val_metrics, compute_fiq_val_metrics


def blip_text_finetune_fiq(train_dress_types: List[str], val_dress_types: List[str],
                            num_epochs: int, batch_size: int,
                            blip_pretrained_path: str, med_config_path: str,
                            blip_learning_rate: float, blip_min_lr: float, blip_max_epoch: int, 
                            validation_frequency: int, transform: str, input_dim: int,
                            save_training: bool, save_best: bool,
                            **kwargs):
    """
    Fine-tune BLIP text encoder on the FashionIQ dataset using as combining function the image-text element-wise sum
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on

    :param num_epochs: number of epochs
    :param batch_size: batch size
    :param blip_learning_rate: fine-tuning learning rate
    :param blip_min_lr: minimum learning rate for cosine learning rate scheduler
    :param blip_max_epoch: maximum training epochs for cosine learning rate scheduler
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the fine-tuned BLIP model
    :param save_best: when True save only the weights of the best BLIP model wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg
    """
    encoder = 'text' # we only finetune BLIP text encoder
    grad_accumulation_step = 1 # gradient accumulation, though we have not used it

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/blip_text_finetuned_on_fiq_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"training start time {training_start}")
    print(f"local folder {training_path}")

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    from blip_modules.blip_text_encoder import BLIPTextEncoder
    blip_text_encoder = BLIPTextEncoder(blip_pretrained_path, med_config_path, use_pretrained_proj_layer=True) # create BLIP text encoder, load pre-trained checkpoint
    blip_text_encoder = blip_text_encoder.to(device)
    print("blip text encoder loaded.")
    blip_text_encoder.eval()

    from blip_modules.blip_img_encoder import BLIPImgEncoder
    blip_img_encoder = BLIPImgEncoder(blip_pretrained_path) # create BLIP text encoder, load pre-trained checkpoint
    blip_img_encoder = blip_img_encoder.to(device)
    print("blip img encoder loaded.")
    blip_img_encoder.eval()

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs
    if encoder == 'text':
        index_features_list = []
        index_names_list = []

    # Define the validation datasets
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, )
        classic_val_datasets.append(classic_val_dataset)
        if encoder == 'text':
            index_features_and_names = extract_index_features(classic_val_dataset, blip_img_encoder)
            index_features_list.append(index_features_and_names[0])
            index_names_list.append(index_features_and_names[1])

    # Define the train datasets and the combining function
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=8, pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)
    combining_function = element_wise_sum

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': [param for name, param in blip_text_encoder.named_parameters()
                        if 'text_proj' not in name], 'lr': blip_learning_rate,
          'weight_decay': 0.05},
        {'params': [param for name, param in blip_text_encoder.named_parameters()
                        if 'text_proj' in name], 'lr': blip_learning_rate * 100, # use larger lr for text_proj layer
          'weight_decay': 0.05}])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        with experiment.train():
            blip_text_encoder.train()
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(relative_train_loader, ncols=150)

            cosine_lr_schedule(optimizer, epoch, blip_max_epoch, blip_learning_rate, blip_min_lr, onlyGroup0=True)

            for idx, (reference_images, target_images, captions, negated_captions) in enumerate(train_bar):
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx
                
                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                with torch.no_grad():
                    reference_features = blip_img_encoder(reference_images)
                    target_features = F.normalize(blip_img_encoder(target_images))

                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                flattened_captions: list = np.array(captions).T.flatten().tolist()
                flattened_negated_captions: list = np.array(negated_captions).T.flatten().tolist()

                captions = generate_randomized_fiq_caption(flattened_captions)
                negated_captions = generate_randomized_fiq_caption(flattened_negated_captions)

                # Extract the features, compute the logits and the loss
                with torch.cuda.amp.autocast():
                    caption_features = blip_text_encoder(captions, max_length=77, device=device)
                    negated_caption_features = blip_text_encoder(negated_captions, max_length=77, device=device)

                    # forward queries
                    predicted_features = element_wise_sum(reference_features, caption_features)

                    logits = 100 * predicted_features @ target_features.T

                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss_f = crossentropy_criterion(logits, ground_truth)

                    # reversed queries
                    '''
                    manually construct the loss logits
                    construct a Tij matrix of dimension B x B x dim
                    a ti tensor of B x dim
                    an Ri tensor of B x dim
                    '''
                    Tij = target_features.unsqueeze(1).repeat(1, images_in_batch, 1).transpose(0,1)
                    ti = negated_caption_features.unsqueeze(1).repeat(1, images_in_batch, 1)
                    Ri = F.normalize(reference_features, dim=-1).unsqueeze(1).repeat(1, images_in_batch, 1).unsqueeze(-1)
                    Pi = element_wise_sum(Tij, ti).unsqueeze(-2)

                    logits_r = 100 * Pi @ Ri
                    logits_r = logits_r.squeeze() # B x B

                    loss_r = crossentropy_criterion(logits_r, ground_truth)

                    loss = loss_f + .4 * loss_r

                loss /= grad_accumulation_step
                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                if ((idx + 1) % grad_accumulation_step == 0) or (idx + 1 == len(relative_train_loader)):
                    scaler.step(optimizer)
                    scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                experiment.log_metric('step_loss_forward', loss_f.detach().cpu().item(), step=step)
                experiment.log_metric('step_loss_reversed', loss_r.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            with experiment.validate():
                blip_text_encoder.eval()
                recalls_at10 = []
                recalls_at50 = []

                for relative_val_dataset, classic_val_dataset, idx in zip(relative_val_datasets, classic_val_datasets,
                                                                          idx_to_dress_mapping):
                    if encoder == 'text': # should have been precomputed if only fine-tuning on text
                        index_features, index_names = index_features_list[idx], index_names_list[idx]
                    else:
                        index_features, index_names = extract_index_features(classic_val_dataset, blip_img_encoder)

                    recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, 
                                                                       blip_text_encoder,
                                                                       index_features, 
                                                                       index_names,
                                                                       combining_function)
                    recalls_at10.append(recall_at10)
                    recalls_at50.append(recall_at50)

                results_dict = {}
                for i in range(len(recalls_at10)):
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict.update({
                    f'average_recall_at10': mean(recalls_at10),
                    f'average_recall_at50': mean(recalls_at50),
                    f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
                })

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            if save_training:
                save_model('tuned_blip_last', epoch, blip_text_encoder, training_path, optimizer)

                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('tuned_blip_best', epoch, blip_text_encoder, training_path, optimizer)
                elif not save_best:
                    save_model(f'tuned_blip_{epoch}', epoch, blip_text_encoder, training_path, optimizer)


def blip_text_finetune_cirr(num_epochs: int, batch_size: int,
                            blip_pretrained_path: str, med_config_path: str,
                            blip_learning_rate: float, blip_min_lr: float, blip_max_epoch: int,
                            validation_frequency: int, transform: str, input_dim: int,
                            save_training: bool, save_best: bool,
                            **kwargs):
    """
    Fine-tune BLIP text encoder on the CIRR dataset using as combining function the image-text element-wise sum
    :param num_epochs: number of epochs
    :param batch_size: batch size
    :param blip_learning_rate: fine-tuning learning rate
    :param blip_min_lr: minimum learning rate for cosine learning rate scheduler
    :param blip_max_epoch: maximum training epochs for cosine learning rate scheduler
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    :return:
    """
    encoder = 'text' # we only finetune BLIP text encoder
    grad_accumulation_step = 1 # gradient accumulation, though we have not used it

    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/blip_text_finetuned_on_cirr_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"training start time {training_start}")
    print(f"local folder {training_path}")

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    from blip_modules.blip_text_encoder import BLIPTextEncoder
    blip_text_encoder = BLIPTextEncoder(blip_pretrained_path, med_config_path, use_pretrained_proj_layer=True) # create BLIP text encoder, load pre-trained checkpoint
    blip_text_encoder = blip_text_encoder.to(device)
    print("blip text encoder loaded.")
    blip_text_encoder.eval()

    from blip_modules.blip_img_encoder import BLIPImgEncoder
    blip_img_encoder = BLIPImgEncoder(blip_pretrained_path) # create BLIP text encoder, load pre-trained checkpoint
    blip_img_encoder = blip_img_encoder.to(device)
    print("blip img encoder loaded.")
    blip_img_encoder.eval()

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")

    # Define the validation datasets
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess)
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess)

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over
    # the epochs
    if encoder == 'text':
        val_index_features, val_index_names = extract_index_features(classic_val_dataset, blip_img_encoder)

    # Define the train dataset and the combining function
    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=8, pin_memory=False, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)
    combining_function = element_wise_sum

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(
        [{'params': [param for name, param in blip_text_encoder.named_parameters()
                        if 'text_proj' not in name], 'lr': blip_learning_rate,
          'weight_decay': 0.05},
        {'params': [param for name, param in blip_text_encoder.named_parameters()
                        if 'text_proj' in name], 'lr': blip_learning_rate * 100, # use larger lr for text_proj layer
          'weight_decay': 0.05}])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best results to zero
    if save_best:
        best_harmonic = 0
        best_geometric = 0
        best_arithmetic = 0
        best_mean = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    print('Training loop started')
    for epoch in range(num_epochs):
        with experiment.train():
            blip_text_encoder.train()
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            train_bar = tqdm(relative_train_loader, ncols=150)

            cosine_lr_schedule(optimizer, epoch, blip_max_epoch, blip_learning_rate, blip_min_lr, onlyGroup0=True)

            for idx, (reference_images, target_images, captions, negated_captions) in enumerate(train_bar):
                images_in_batch = reference_images.size(0)
                step = len(train_bar) * epoch + idx

                optimizer.zero_grad()

                reference_images = reference_images.to(device, non_blocking=True)
                target_images = target_images.to(device, non_blocking=True)

                with torch.no_grad():
                    reference_features = blip_img_encoder(reference_images)
                    target_features = F.normalize(blip_img_encoder(target_images), dim=-1)

                with torch.cuda.amp.autocast():
                    text_features = blip_text_encoder(captions, max_length=77, device=device)
                    negated_text_features = blip_text_encoder(negated_captions, max_length=77, device=device)
                        
                    # forward queries
                    predicted_features = combining_function(reference_features, text_features)

                    logits = 100 * predicted_features @ target_features.T

                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss_f = crossentropy_criterion(logits, ground_truth)

                    # reversed queries
                    '''
                    manually construct the loss logits
                    construct a Tij matrix of dimension B x B x dim
                    a ti tensor of B x dim
                    an Ri tensor of B x dim
                    '''
                    Tij = target_features.unsqueeze(1).repeat(1, images_in_batch, 1).transpose(0,1)
                    ti = negated_text_features.unsqueeze(1).repeat(1, images_in_batch, 1)
                    Ri = F.normalize(reference_features, dim=-1).unsqueeze(1).repeat(1, images_in_batch, 1).unsqueeze(-1)
                    Pi = combining_function(Tij, ti).unsqueeze(-2)

                    logits_r = 100 * Pi @ Ri
                    logits_r = logits_r.squeeze() # B x B

                    loss_r = crossentropy_criterion(logits_r, ground_truth)

                    loss = loss_f + .1 * loss_r

                loss /= grad_accumulation_step
                # Backpropagate and update the weights
                scaler.scale(loss).backward()
                if ((idx + 1) % grad_accumulation_step == 0) or (idx + 1 == len(relative_train_loader)):
                    scaler.step(optimizer)
                    scaler.update()

                    experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                    experiment.log_metric('step_loss_forward', loss_f.detach().cpu().item(), step=step)
                    experiment.log_metric('step_loss_reversed', loss_r.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            with experiment.validate():
                blip_text_encoder.eval()
                if encoder != 'text': # should have been precomputed if only fine-tuning on text
                    val_index_features, val_index_names = extract_index_features(classic_val_dataset, blip_img_encoder)

                results = compute_cirr_val_metrics(relative_val_dataset, 
                                                   blip_text_encoder, 
                                                   val_index_features, 
                                                   val_index_names, 
                                                   combining_function)
                group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results

                results_dict = {
                    'group_recall_at1': group_recall_at1,
                    'group_recall_at2': group_recall_at2,
                    'group_recall_at3': group_recall_at3,
                    'recall_at1': recall_at1,
                    'recall_at5': recall_at5,
                    'recall_at10': recall_at10,
                    'recall_at50': recall_at50,
                    'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                    'arithmetic_mean': mean(results),
                    'harmonic_mean': harmonic_mean(results),
                    'geometric_mean': geometric_mean(results)
                }
                print(json.dumps(results_dict, indent=4))

                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                if save_training:
                    save_model('tuned_blip_last', epoch, blip_text_encoder, training_path, optimizer)
                    if save_best and results_dict['mean(R@5+R_s@1)'] > best_mean:
                        best_mean = results_dict['mean(R@5+R_s@1)']
                        save_model('tuned_blip_mean', epoch, blip_text_encoder, training_path, optimizer)
                    if save_best and results_dict['arithmetic_mean'] > best_arithmetic:
                        best_arithmetic = results_dict['arithmetic_mean']
                        save_model('tuned_blip_arithmetic', epoch, blip_text_encoder, training_path, optimizer)
                    if save_best and results_dict['harmonic_mean'] > best_harmonic:
                        best_harmonic = results_dict['harmonic_mean']
                        save_model('tuned_blip_harmonic', epoch, blip_text_encoder, training_path, optimizer)
                    if save_best and results_dict['geometric_mean'] > best_geometric:
                        best_geometric = results_dict['geometric_mean']
                        save_model('tuned_blip_geometric', epoch, blip_text_encoder, training_path, optimizer)
                    if not save_best:
                        save_model(f'tuned_blip_{epoch}', epoch, blip_text_encoder, training_path, optimizer)


if __name__ == '__main__':
    parser = ArgumentParser()
    # dataset
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    # comet environment
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    # BLIP pretrain
    parser.add_argument("--blip-pretrained-path", default='models/model_base.pth', type=str, help="path of the BLIP pretrained model weights")
    parser.add_argument("--med-config-path", default='src/blip_modules/med_config.json', type=str, help="path of the BLIP text encoder med_config.json")
    # training args
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--blip-learning-rate", default=1e-5, type=float, help="BLIP text encoder learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    # cosine learning rate scheduler
    parser.add_argument("--blip-min-lr", default=0, type=float, help="Cos Learning Rate Scheduler min learning rate")
    parser.add_argument("--blip-max-epoch", default=10, type=int, help="Cos Learning Rate Scheduler max epoch")
    # image preprocessing
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['squarepad', 'targetpad'] ")
    parser.add_argument("--input-dim", default=384, type=int, help="Input dimension for image transform. Default: inherited from clip_model.visual.input_resolution")
    # training settings
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "blip_pretrained_path": args.blip_pretrained_path,
        "med_config_path": args.med_config_path,
        "blip_learning_rate": args.blip_learning_rate,
        "blip_max_epoch": args.blip_max_epoch,
        "blip_min_lr": args.blip_min_lr,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "input_dim": args.input_dim,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"BLIP4Cir_Bi blip_text_finetune {args.dataset}",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)
    
    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    import numpy as np
    np.random.seed(random_seed)
    
    if args.dataset.lower() == 'cirr':
        blip_text_finetune_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        blip_text_finetune_fiq(**training_hyper_params)
