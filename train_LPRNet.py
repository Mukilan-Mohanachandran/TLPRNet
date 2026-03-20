# -*- coding: utf-8 -*-

"""
LPRNet training script — faithful to arXiv:1806.10447 Sec. 3.2.

Key paper settings:
  - Adam optimizer, lr=0.001
  - Batch size 32
  - LR drops by 10x every 100k iterations, 250k total
  - Gradient noise scale 0.001
  - Random affine data augmentation
  - Optional STN with 5k-iteration warmup
"""

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os


def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int64)
    return torch.stack(imgs, 0), torch.from_numpy(labels), lengths


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def get_parser():
    parser = argparse.ArgumentParser(description='LPRNet Training (paper-faithful)')
    parser.add_argument('--img_size', default=[94, 24], help='input image size [W, H]')
    parser.add_argument('--train_img_dirs', default="~/workspace/trainMixLPR",
                        help='comma-separated training image directories')
    parser.add_argument('--test_img_dirs', default="~/workspace/testMixLPR",
                        help='comma-separated test image directories')
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='initial learning rate (paper: 0.001)')
    parser.add_argument('--lpr_max_len', default=8, type=int,
                        help='max characters on a license plate')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='training batch size (paper: 32)')
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_iter', default=250000, type=int,
                        help='total training iterations (paper: 250k)')
    parser.add_argument('--lr_step', default=100000, type=int,
                        help='drop LR every N iters (paper: 100k)')
    parser.add_argument('--gradient_noise', default=0.001, type=float,
                        help='gradient noise scale (paper: 0.001)')
    parser.add_argument('--use_stn', action='store_true',
                        help='enable Spatial Transformer Network')
    parser.add_argument('--stn_warmup', default=5000, type=int,
                        help='iterations before enabling STN (paper: 5k)')
    parser.add_argument('--backbone_type', default='lprnet',
                        choices=['lprnet', 'svtr_lcnet'],
                        help='recognizer backbone variant')
    parser.add_argument('--mix_dim', default=192, type=int,
                        help='hybrid backbone token/channel dimension')
    parser.add_argument('--mix_blocks', default=2, type=int,
                        help='number of global mix blocks for hybrid backbone')
    parser.add_argument('--mix_heads', default=4, type=int,
                        help='attention heads in each global mix block')
    parser.add_argument('--augment', action='store_true',
                        help='enable random affine augmentation (paper: yes)')
    parser.add_argument('--cuda', default=True, type=str2bool)
    parser.add_argument('--resume_iter', default=0, type=int,
                        help='resume from this iteration')
    parser.add_argument('--save_interval', default=5000, type=int)
    parser.add_argument('--test_interval', default=5000, type=int)
    parser.add_argument('--save_folder', default='./weights/')
    parser.add_argument('--pretrained_model', default='',
                        help='path to pretrained model weights')
    return parser.parse_args()


def train():
    args = get_parser()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    lprnet = build_lprnet(
        class_num=len(CHARS),
        dropout_rate=args.dropout_rate,
        use_stn=args.use_stn,
        training=True,
        backbone_type=args.backbone_type,
        mix_dim=args.mix_dim,
        mix_blocks=args.mix_blocks,
        mix_heads=args.mix_heads,
    )
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successfully built LPRNet!")

    if args.pretrained_model:
        lprnet.load_state_dict(
            torch.load(args.pretrained_model, map_location=device)
        )
        print("Loaded pretrained model:", args.pretrained_model)

    # Paper Sec. 3.2: Adam, lr=0.001
    optimizer = optim.Adam(lprnet.parameters(), lr=args.learning_rate)

    # Paper Sec. 3.2: drop LR by 10x every 100k iterations
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=0.1
    )

    train_dataset = LPRDataLoader(
        os.path.expanduser(args.train_img_dirs).split(','),
        args.img_size, args.lpr_max_len, augment=args.augment,
    )
    test_dataset = LPRDataLoader(
        os.path.expanduser(args.test_img_dirs).split(','),
        args.img_size, args.lpr_max_len, augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )

    ctc_loss = nn.CTCLoss(blank=len(CHARS) - 1, reduction='mean')

    # Paper Sec. 3.2: STN warmup — disable for first 5k iterations
    if args.use_stn:
        lprnet.stn_enabled = False

    train_iter = iter(train_loader)

    for iteration in range(args.resume_iter, args.max_iter):
        # Reload data loader when exhausted
        try:
            images, labels, lengths = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels, lengths = next(train_iter)

        # Enable STN after warmup period
        if args.use_stn and iteration == args.stn_warmup:
            lprnet.stn_enabled = True
            print(f"STN enabled at iteration {iteration}")

        images = images.to(device)
        labels = labels.to(device)

        start_time = time.time()

        # Forward
        logits = lprnet(images)
        T_length = logits.size(2)
        log_probs = logits.permute(2, 0, 1).log_softmax(2)

        input_lengths = torch.full(
            (images.size(0),), T_length, dtype=torch.long
        )
        target_lengths = torch.tensor(lengths, dtype=torch.long)

        # Backward
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
        if loss.item() == np.inf:
            continue
        loss.backward()

        # Paper Sec. 3.2: gradient noise
        if args.gradient_noise > 0:
            for p in lprnet.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * args.gradient_noise)

        optimizer.step()
        scheduler.step()

        batch_time = time.time() - start_time

        if iteration % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f"Iter {iteration}/{args.max_iter} | "
                f"Loss: {loss.item():.4f} | "
                f"Batch: {batch_time:.4f}s | "
                f"LR: {lr:.8f}"
            )

        if iteration != 0 and iteration % args.save_interval == 0:
            path = os.path.join(
                args.save_folder, f'LPRNet_iter_{iteration}.pth'
            )
            torch.save(lprnet.state_dict(), path)

        if iteration != 0 and iteration % args.test_interval == 0:
            greedy_decode_eval(lprnet, test_dataset, args)

    # Final evaluation and save
    print("Final test accuracy:")
    greedy_decode_eval(lprnet, test_dataset, args)

    final_path = os.path.join(args.save_folder, 'Final_LPRNet_model.pth')
    torch.save(lprnet.state_dict(), final_path)
    print("Training complete. Model saved to", final_path)


def greedy_decode_eval(net, dataset, args):
    """Evaluate with CTC greedy decoding."""
    net.eval()
    device = next(net.parameters()).device
    loader = DataLoader(
        dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    blank = len(CHARS) - 1
    tp, tn_len, tn_char = 0, 0, 0
    t_start = time.time()

    with torch.no_grad():
        for images, labels, lengths in loader:
            images = images.to(device)
            logits = net(images).cpu().numpy()

            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length

            for i in range(logits.shape[0]):
                pred = logits[i]  # (class_num, T)
                indices = pred.argmax(axis=0)

                collapsed = []
                prev = -1
                for idx in indices:
                    if idx != prev:
                        collapsed.append(idx)
                    prev = idx
                decoded = [c for c in collapsed if c != blank]

                if len(decoded) != len(targets[i]):
                    tn_len += 1
                elif (np.array(decoded) == targets[i]).all():
                    tp += 1
                else:
                    tn_char += 1

    total = tp + tn_len + tn_char
    acc = tp / total if total > 0 else 0
    elapsed = time.time() - t_start
    print(
        f"[Test] Accuracy: {acc:.4f} "
        f"[{tp}:{tn_len}:{tn_char}:{total}] "
        f"Speed: {elapsed / max(len(dataset), 1):.4f}s/img"
    )
    net.train()


if __name__ == "__main__":
    train()
