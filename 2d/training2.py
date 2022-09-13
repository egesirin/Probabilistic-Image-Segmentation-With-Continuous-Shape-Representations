import torch
import math
from mc_samples import mc_sample_cov_is_low_rank
from metrics2 import gen_energy_distance
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_low_rank(epoch, model, loss_function, optimizer, number_of_samples, coords, training_set, validation_set,
                   batch_size, writer):
    model.train()
    print("Train/epoch : ", epoch)
    global_step = 0
    for idx, (image, gts, uid) in enumerate(iter(training_set)):
        image = image.to(device=DEVICE)
        gts = torch.stack(gts)
        gts = gts.reshape(len(gts), batch_size, 16384)
        gts = gts.permute(1, 0, 2).to(device=DEVICE)
        mean_vector = model(coords, image)[0]
        low_rank_factor = model(coords, image)[1]
        log_prob = torch.zeros(batch_size, 4, number_of_samples).to(device=DEVICE)
        #losses = torch.zeros(batch_size)
        #print(idx)
        for i in range(batch_size):
            for j in range(gts.shape[1]):
                gt = gts[i][j]
                #correct but not good!
                mc_sample = mc_sample_cov_is_low_rank(mean_vector, low_rank_factor, number_of_samples)[i].to(
                    device=DEVICE)
                for k in range(number_of_samples):
                    log_prob[i][j][k] = -loss_function(mc_sample[k], gt)
            #losses[i] = torch.mean(-torch.logsumexp(log_prob[i], dim=2) + math.log(number_of_samples))
        #loss = torch.mean(losses)
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
        global_step += 1
        if global_step % 20 == 0:
            print(global_step)
            writer.add_scalar('Training/Loss', loss.item(), global_step=global_step)
            writer.add_scalar('Training/Ged', ged.item(), global_step=global_step)
    print("Validation/epoch : ", epoch)
    #model.eval()
    with torch.no_grad():
        for idx, (image, gts, uid) in enumerate(iter(validation_set)):
            image = image.to(device=DEVICE)
            gts = torch.stack(gts)
            gts = gts.reshape(len(gts), batch_size, 16384)
            gts = gts.permute(1, 0, 2).to(device=DEVICE)
            mean_vector = model(coords, image)[0]
            low_rank_factor = model(coords, image)[1]
            log_prob = torch.zeros(batch_size, 4, number_of_samples).to(device=DEVICE)
            #losses = torch.zeros(batch_size)
            #print(idx)
            for i in range(batch_size):
                for j in range(gts.shape[1]):
                    gt = gts[i][j]
                    # correct but not good!
                    mc_sample = mc_sample_cov_is_low_rank(mean_vector, low_rank_factor, number_of_samples)[i].to(
                        device=DEVICE)
                    for k in range(number_of_samples):
                        log_prob[i][j][k] = -loss_function(mc_sample[k], gt)
                        # print(log_prob)
                #losses[i] = torch.mean(-torch.logsumexp(log_prob[i], dim=2) + math.log(number_of_samples))
            #loss = torch.mean(losses)
            batch_losses = torch.mean(-torch.logsumexp(log_prob, dim=2) + math.log(number_of_samples), dim=1)
            loss = torch.mean(batch_losses)
            cross, gts_diversity, sample_diversity = gen_energy_distance(image, gts, model, coords, 100, DEVICE, batch_size)
            ged = 2 * cross - gts_diversity - sample_diversity
            del mc_sample
            del gts
            del image
            if global_step % 20 == 0:
                print(global_step)
                writer.add_scalar('Validation/Loss', loss.item(), global_step=global_step)
                writer.add_scalar('Validation/Ged', ged.item(), global_step=global_step)

