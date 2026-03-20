# -*- coding: utf-8 -*-

"""
LPRNet inference / evaluation script.

Supports greedy and beam-search CTC decoding (Sec. 3.1).
"""

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from model.LPRNet import build_lprnet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import torch
import time
import cv2
import os


# ---------------------------------------------------------------------------
# CTC decoders
# ---------------------------------------------------------------------------

def greedy_decode(logits, blank):
    """CTC greedy decoding for a single sample.

    Args:
        logits: (C, T) raw logits
        blank:  index of the CTC blank token
    Returns:
        list of decoded token indices
    """
    indices = logits.argmax(axis=0)
    collapsed = []
    prev = -1
    for idx in indices:
        if idx != prev:
            collapsed.append(idx)
        prev = idx
    return [c for c in collapsed if c != blank]


def beam_search_decode(logits, beam_width=10, blank=0):
    """CTC prefix beam search for a single sample.

    Tracks (prob_ending_in_blank, prob_ending_in_non_blank) per prefix
    to correctly handle repeated characters separated by blanks.

    Args:
        logits:     (C, T) raw logits
        beam_width: number of hypotheses to keep
        blank:      index of the CTC blank token
    Returns:
        list of decoded token indices for the best beam
    """
    C, T = logits.shape
    log_probs = torch.log_softmax(torch.tensor(logits), dim=0).numpy()
    NEG_INF = float('-inf')

    # {prefix_tuple: (log_prob_blank, log_prob_non_blank)}
    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        new_beams = {}

        def _add(prefix, pb, pnb):
            if prefix in new_beams:
                opb, opnb = new_beams[prefix]
                new_beams[prefix] = (
                    np.logaddexp(opb, pb),
                    np.logaddexp(opnb, pnb),
                )
            else:
                new_beams[prefix] = (pb, pnb)

        for prefix, (pb, pnb) in beams.items():
            p_total = np.logaddexp(pb, pnb)

            # Blank extension
            _add(prefix, p_total + log_probs[blank, t], NEG_INF)

            # Character extensions
            for c in range(C):
                if c == blank:
                    continue
                if len(prefix) > 0 and prefix[-1] == c:
                    # Repeated char: only extend after blank (new char)
                    _add(prefix + (c,), NEG_INF, pb + log_probs[c, t])
                    # Collapse (same prefix kept): after non-blank
                    _add(prefix, NEG_INF, pnb + log_probs[c, t])
                else:
                    _add(prefix + (c,), NEG_INF, p_total + log_probs[c, t])

        # Prune to beam_width
        scored = sorted(
            new_beams.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1]),
            reverse=True,
        )
        beams = dict(scored[:beam_width])

    best = max(
        beams.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1])
    )
    return list(best[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    if v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def get_parser():
    parser = argparse.ArgumentParser(description='LPRNet Testing')
    parser.add_argument('--img_size', default=[94, 24])
    parser.add_argument('--test_img_dirs', default="./data/test")
    parser.add_argument('--dropout_rate', default=0, type=float)
    parser.add_argument('--lpr_max_len', default=8, type=int)
    parser.add_argument('--test_batch_size', default=100, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cuda', default=True, type=str2bool)
    parser.add_argument('--show', default=False, type=str2bool,
                        help='display predictions visually')
    parser.add_argument('--use_stn', action='store_true',
                        help='must match training flag')
    parser.add_argument('--backbone_type', default='lprnet',
                        choices=['lprnet', 'svtr_lcnet'],
                        help='must match training backbone')
    parser.add_argument('--mix_dim', default=192, type=int,
                        help='hybrid backbone token/channel dimension')
    parser.add_argument('--mix_blocks', default=2, type=int,
                        help='number of global mix blocks for hybrid backbone')
    parser.add_argument('--mix_heads', default=4, type=int,
                        help='attention heads in each global mix block')
    parser.add_argument('--pretrained_model',
                        default='./weights/Final_LPRNet_model.pth')
    parser.add_argument('--decode_method', default='greedy',
                        choices=['greedy', 'beam_search'],
                        help='CTC decoding strategy (Sec. 3.1)')
    parser.add_argument('--beam_width', default=10, type=int,
                        help='beam width for beam_search decode')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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


def test():
    args = get_parser()

    lprnet = build_lprnet(
        class_num=len(CHARS),
        dropout_rate=args.dropout_rate,
        use_stn=args.use_stn,
        training=False,
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
    else:
        print("[Error] No pretrained model specified!")
        return

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset = LPRDataLoader(
        test_img_dirs.split(','), args.img_size, args.lpr_max_len
    )
    try:
        evaluate(lprnet, test_dataset, args, device)
    finally:
        cv2.destroyAllWindows()


def evaluate(net, dataset, args, device):
    loader = DataLoader(
        dataset, batch_size=args.test_batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    blank = len(CHARS) - 1
    tp, tn_len, tn_char = 0, 0, 0
    t_start = time.time()

    with torch.no_grad():
        for images, labels, lengths in loader:
            start = 0
            targets = []
            for length in lengths:
                targets.append(labels[start:start + length].numpy())
                start += length

            imgs_np = images.numpy().copy()
            images = images.to(device)
            logits = net(images).cpu().numpy()

            for i in range(logits.shape[0]):
                pred = logits[i]  # (class_num, T)

                if args.decode_method == 'beam_search':
                    decoded = beam_search_decode(
                        pred, beam_width=args.beam_width, blank=blank
                    )
                else:
                    decoded = greedy_decode(pred, blank)

                if args.show:
                    show(imgs_np[i], decoded, targets[i])

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
        f"[{tp}:{tn_len}:{tn_char}:{total}]"
    )
    print(
        f"[Test] Speed: {elapsed / max(len(dataset), 1):.4f}s/img "
        f"({len(dataset)} images)"
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def show(img, label, target):
    img = np.transpose(img, (1, 2, 0))
    img *= 128.0
    img += 127.5
    img = img.astype(np.uint8)

    lb = "".join(CHARS[i] for i in label)
    tg = "".join(CHARS[int(j)] for j in target)

    flag = "T" if lb == tg else "F"
    img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3),
                     interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, lb, (2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("test", img)
    print(f"target: {tg}  ### {flag} ###  predict: {lb}")
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
