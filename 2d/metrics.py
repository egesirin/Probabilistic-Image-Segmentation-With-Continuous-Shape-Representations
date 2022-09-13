import torch
from mc_samples import mc_sample_cov_is_low_rank


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


def gen_energy_distance(im, gtss, model, coords, sample_num, DEVICE, batch):
    with torch.no_grad():
        gtss = gtss.to(dtype=torch.long)
        mean_vector = model(coords, im)[0]
        low_rank_factor = model(coords, im)[1]
        samples = mc_sample_cov_is_low_rank(mean_vector, low_rank_factor, sample_num).to(device=DEVICE)
        samples = torch.round(torch.sigmoid(samples)).to(dtype=torch.long)
        eye = torch.eye(2)
        gt_dist = eye[gtss].to(dtype=bool)
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
