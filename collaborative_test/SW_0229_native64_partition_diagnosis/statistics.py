"""Pair-error accounting and visible topology; no predicted masks are changed."""
import numpy as np
from scipy import ndimage

FOUR = ndimage.generate_binary_structure(2, 1)


def choose2(values):
    values = np.asarray(values, dtype=np.int64)
    return values * (values - 1) // 2


def labels2d(value):
    value = np.asarray(value)
    if value.ndim != 2 or value.dtype.kind not in 'iu' or (value < 0).any():
        raise ValueError('nonnegative integer label grid required')
    return value


def pair_errors(target, prediction):
    target = labels2d(target); prediction = labels2d(prediction)
    if target.shape != prediction.shape:
        raise ValueError('same patch geometry required')
    foreground = target > 0
    truth, ti = np.unique(target[foreground], return_inverse=True)
    predicted, pi = np.unique(prediction[foreground], return_inverse=True)
    table = np.zeros((len(truth), len(predicted)), dtype=np.int64)
    np.add.at(table, (ti, pi), 1)
    gt_counts = table.sum(1); pred_counts = table.sum(0)
    true_pairs = int(choose2(table).sum()); same_gt = int(choose2(gt_counts).sum())
    merge = int(choose2(pred_counts).sum()) - true_pairs
    split = same_gt - true_pairs
    bg_column = np.flatnonzero(predicted == 0)
    bg_counts = table[:, int(bg_column[0])] if len(bg_column) else np.zeros(len(truth), dtype=np.int64)
    merge_bg = int(choose2(bg_counts.sum()) - choose2(bg_counts).sum())
    split_support = int((bg_counts * (gt_counts - bg_counts)).sum())
    different_gt = int(choose2(int(foreground.sum()))) - same_gt
    overlap = int((foreground & (prediction > 0)).sum())
    result = dict(true_positive_pairs=true_pairs, GT_same_pairs=same_gt, GT_different_pairs=different_gt,
        false_merge_pairs=merge, false_merge_background_pairs=merge_bg, false_merge_foreground_pairs=merge - merge_bg,
        false_split_pairs=split, false_split_support_pairs=split_support, false_split_inside_foreground_pairs=split - split_support,
        false_merge_rate=merge / max(1, different_gt), false_split_rate=split / max(1, same_gt),
        foreground_precision=overlap / max(1, int((prediction > 0).sum())),
        foreground_recall=overlap / max(1, int(foreground.sum())),
        GT_objects=len(truth), GT_objects_all_predicted_background=int(np.count_nonzero(bg_counts == gt_counts)),
        GT_all_background_object_fraction=int(np.count_nonzero(bg_counts == gt_counts)) / max(1, len(truth)))
    if any(result[k] < 0 for k in result):
        raise ValueError('negative pair accounting')
    return result


def visible_geometry(labels):
    labels = labels2d(labels)
    ids = np.unique(labels); ids = ids[ids > 0]
    multipart = components = minor = 0
    for key in ids:
        mask = labels == key
        separated, count = ndimage.label(mask, structure=FOUR)
        sizes = np.bincount(separated[mask])[1:]
        multipart += int(count > 1); components += count
        minor += int(mask.sum()) - int(sizes.max())
    return dict(objects=len(ids), multipart_objects=multipart, visible_components=components,
                minor_component_patches=minor, foreground_patches=int((labels > 0).sum()))
