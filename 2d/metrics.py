import torch
import math
from mc_samples import sampling


def dice_score(gts, samples, axis=-1, ignore_empty_slices: bool = True):
    """Return the sum of dice scores in this batch."""
    samples = torch.round(torch.sigmoid(samples)).to(dtype=torch.long)
    gts = gts.to(dtype=torch.long)
    dice_ = (2*(gts & samples).sum(axis)) / (gts.sum(axis) + samples.sum(axis))
    dice_[torch.isnan(dice_)] = 1.  # [batch, num_mc_samples]
    if ignore_empty_slices:
        are_slices_empty = torch.all(gts == 0, dim=axis)  # [batch, num_mc_samples]
        are_slices_not_empty = torch.logical_not(are_slices_empty)  # 1 = not-empty, 0 = empty
        num_slices_considered = are_slices_not_empty.sum()
        dice = (dice_ * are_slices_not_empty).sum()
    else:
        num_slices_considered = dice_.numel()
        dice = torch.sum(dice_)
    return dice, num_slices_considered


def iou(x, y, axis=-1):
    iou_ = (x & y).sum(axis) / (x | y).sum(axis)
    iou_[torch.isnan(iou_)] = 1.
    return iou_


def distance(x, y):
    try:
        per_class_iou = iou(x[:, None], y[None, :], axis=-2)
    except MemoryError:
        per_class_iou = []
        for x_ in x:
            per_class_iou.append(iou(torch.unsqueeze(x_, dim=0), y[None, :], axis=-2))
        per_class_iou = torch.cat(per_class_iou)
    return 1 - per_class_iou[..., 1:].mean(-1)


def gen_energy_distance(gts, samples, batch):
    gts = gts.to(dtype=torch.long)
    samples = torch.round(torch.sigmoid(samples)).to(dtype=torch.long)
    eye = torch.eye(2)
    gt_dist = eye[gts].to(dtype=bool)
    sample_dist = eye[samples].to(dtype=bool)
    cross = torch.zeros(batch)
    gts_diversity = torch.zeros(batch)
    sample_diversity = torch.zeros(batch)
    for i in range(batch):
        gts_diversity[i] = torch.mean(distance(gt_dist[i], gt_dist[i]))
        sample_diversity[i] = torch.mean(distance(sample_dist[i], sample_dist[i]))
        cross[i] = torch.mean(distance(sample_dist[i], gt_dist[i]))
    cross = torch.mean(cross)
    gts_diversity = torch.mean(gts_diversity)
    sample_diversity = torch.mean(sample_diversity)
    del samples
    return cross, gts_diversity, sample_diversity


def loss_func(im, gt, model, model_type, loss_function, number_of_samples_per_gt,
            coords, batch_size, DEVICE):
    image = im.to(device=DEVICE)
    gts = torch.stack(gt)
    gts = gts.reshape(len(gts), batch_size, -1)
    gts = gts.permute(1, 0, 2).to(device=DEVICE)  # torch.Size([batch, #ground_truths, #pixels])
    number_of_gts = gts.size(1)
    num_samples = number_of_gts * number_of_samples_per_gt
    mc_samples = sampling(model(coords, image), model_type, num_samples)
    gts = gts.repeat(1, number_of_samples_per_gt, 1)  # torch.Size([batch, num_samples,  #pixels])
    # sum over the pixels part in equation 7 (SSN) ---> loss reduction=sum
    log_prob = -loss_function(mc_samples, gts).sum(2)  # torch.Size([batch, num_samples]
    # equation 7 in SSN. the formula (7) is only for one gt.
    batch_losses = ((-torch.logsumexp(log_prob, dim=1)) + math.log(number_of_samples_per_gt))
    # Now, I don't have to take mean over the number of ground truths, then over batches. It is the same result!
    loss = torch.mean(batch_losses)  # without for loop <3
    del mc_samples
    del gts
    del image
    return loss


def metrics_val(im, gt, model,  model_type, num_of_val_sample, coords, batch_size, DEVICE):
    image = im.to(device=DEVICE)
    gts = torch.stack(gt)
    gts = gts.reshape(len(gts), batch_size, -1)
    gts = gts.permute(1, 0, 2).to(device=DEVICE)  # torch.Size([batch, #ground_truths, #pixels])
    mc_for_ged = sampling(model(coords, image), model_type, num_of_val_sample)
    mc_for_dice = mc_for_ged[:, :4]
    dice_nod, num_slices_nod = dice_score(gts, mc_for_dice, -1, True)
    dice, num_slices = dice_score(gts, mc_for_dice, -1, False)
    cross, gts_diversity, sample_diversity = gen_energy_distance(gts, mc_for_ged, batch_size)
    ged = 2 * cross - gts_diversity - sample_diversity
    del mc_for_dice
    del mc_for_ged
    del gts
    del image
    return ged, cross, gts_diversity, sample_diversity, dice_nod, num_slices_nod, dice, num_slices
