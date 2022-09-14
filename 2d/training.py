import torch
import math
from mc_samples import mc_sample_cov_is_low_rank
from metrics import gen_energy_distance
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_low_rank(epoch,iterations_per_epoch, model, loss_function, optimizer, number_of_samples, coords, training_set,
                   batch_size, writer):
    model.train()
    print("Train/epoch : ", epoch)
    for idx, (image, gts, uid) in enumerate(iter(training_set)):
        image = image.to(device=DEVICE)
        gts = torch.stack(gts)
        gts = gts.reshape(len(gts), batch_size, 16384)
        gts = gts.permute(1, 0, 2).to(device=DEVICE)
        mean_vector = model(coords, image)[0]
        low_rank_factor = model(coords, image)[1]
        rank = low_rank_factor.size(1)
        log_prob = torch.zeros(batch_size, 4, number_of_samples).to(device=DEVICE)
        for i in range(batch_size):
            mean_vector_i = mean_vector[i]
            low_rank_factor_i = low_rank_factor[i]
            for j in range(gts.shape[1]):
                gt = gts[i][j]
                mc_sample = mc_sample_cov_is_low_rank(mean_vector_i, low_rank_factor_i, rank, number_of_samples).to(
                    device=DEVICE)
                for k in range(number_of_samples):
                    log_prob[i][j][k] = -loss_function(mc_sample[k], gt)
        batch_losses = torch.mean(-torch.logsumexp(log_prob, dim=2) + math.log(number_of_samples), dim=1)
        loss = torch.mean(batch_losses)
        cross, gts_diversity, sample_diversity = gen_energy_distance(image, gts, model, coords, 100, DEVICE, batch_size)
        ged = 2 * cross - gts_diversity - sample_diversity
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del mc_sample
        del gts
        del image
        global_step = (epoch * iterations_per_epoch) + idx
        if global_step % 20 == 0:
            print(epoch, global_step)
            writer.add_scalar('Training/Loss', loss.item(), global_step=global_step)
            writer.add_scalar('Training/Ged', ged.item(), global_step=global_step)



def validation_low_rank(epoch, iterations_per_epoch, model, loss_function, number_of_samples, coords, validation_set,
                        batch_size, writer):
    print("Validation/epoch : ", epoch)
    with torch.no_grad():
        for idx, (image, gts, uid) in enumerate(iter(validation_set)):
            image = image.to(device=DEVICE)
            gts = torch.stack(gts)
            gts = gts.reshape(len(gts), batch_size, 16384)
            gts = gts.permute(1, 0, 2).to(device=DEVICE)
            mean_vector = model(coords, image)[0]
            low_rank_factor = model(coords, image)[1]
            rank = low_rank_factor.size(1)
            log_prob = torch.zeros(batch_size, 4, number_of_samples).to(device=DEVICE)
            for i in range(batch_size):
                mean_vector_i = mean_vector[i]
                low_rank_factor_i = low_rank_factor[i]
                for j in range(gts.shape[1]):
                    gt = gts[i][j]
                    mc_sample = mc_sample_cov_is_low_rank(mean_vector_i, low_rank_factor_i, rank, number_of_samples).to(
                        device=DEVICE)
                    for k in range(number_of_samples):
                        log_prob[i][j][k] = -loss_function(mc_sample[k], gt)
            batch_losses = torch.mean(-torch.logsumexp(log_prob, dim=2) + math.log(number_of_samples), dim=1)
            loss = torch.mean(batch_losses)
            cross, gts_diversity, sample_diversity = gen_energy_distance(image, gts, model, coords, 100, DEVICE, batch_size)
            ged = 2 * cross - gts_diversity - sample_diversity
            del mc_sample
            del gts
            del image
            global_step = (epoch * iterations_per_epoch) + idx
            writer.add_scalar('Validation/Loss', loss.item(), global_step=global_step)
            writer.add_scalar('Validation/Ged', ged.item(), global_step=global_step)
            if global_step % 50 == 0:
                print(epoch, global_step)
