import torch
import math
from mc_samples import sampling


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


def metrics(im, gt, model, model_type, loss_function, number_of_samples_per_gt, num_of_val_sample,
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
    with torch.no_grad():
        mc_for_ged = sampling(model(coords, image), model_type, num_of_val_sample)
    cross, gts_diversity, sample_diversity = gen_energy_distance(gts, mc_for_ged, batch_size)
    ged = 2 * cross - gts_diversity - sample_diversity
    del mc_samples
    del gts
    del image
    return loss, ged, cross, gts_diversity, sample_diversity
