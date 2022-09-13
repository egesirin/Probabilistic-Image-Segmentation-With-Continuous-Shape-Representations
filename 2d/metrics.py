import torch

from mc_samples import mc_sample_cov_is_low_rank, mc_sample_mean


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


def gen_energy_distance(model, coords, sample_num, dataset, DEVICE, batch):
    with torch.no_grad():
        for (image, gts) in iter(dataset):
            image = image.to(device=DEVICE)
            mean_vector = model(coords, image)[0]
            print(1)
            low_rank_factor = model(coords, image)[1]
            print(2)
            samples = mc_sample_cov_is_low_rank(mean_vector, low_rank_factor, sample_num).to(device=DEVICE)
            #print(samples.shape)
            #samples = samples.permute((1, 0, 2))
            samples = torch.round(torch.sigmoid(samples)).to(dtype=torch.long)
            #print(gts[0].shape)
            gts = torch.stack(gts).to(device=DEVICE)
            gts = gts.reshape(len(gts), batch, 16384)
            gts = gts.permute(1, 0, 2).to(dtype=torch.long)
            # check this!
            #gts = gts.reshape(batch, len(gts), -1).to(dtype=torch.long)
            eye = torch.eye(2)
            gt_dist = eye[gts].to(dtype=bool)
            sample_dist = eye[samples].to(dtype=bool)
            cross = 0
            gts_diversity = 0
            sample_diversity = 0
            for i in range(batch):
                gts_diversity += torch.mean(distance(gt_dist[i], gt_dist[i]))
                sample_diversity += torch.mean(distance(sample_dist[i], sample_dist[i]))
                cross += torch.mean(distance(sample_dist[i], gt_dist[i]))
        del samples
        del gts
        return cross/batch, gts_diversity/batch, sample_diversity/batch
