import torch
import math
from mc_samples import mc_sample_cov_is_low_rank
from metrics import gen_energy_distance
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_low_rank(epoch, model, loss_function, optimizer, number_of_samples, coords, training_set, batch):
    model.train()
    log_prob = torch.zeros(batch, 4, number_of_samples).to(device=DEVICE)
    losses = torch.zeros(batch).to(device=DEVICE)
    for (image, gts) in iter(training_set):
        image = image.to(device=DEVICE)
        print(image.shape)
        #gts = gts.to(device=DEVICE)
        mean_vector = model(coords, image)[0]
        #batch = mean_vector.size(0)
        low_rank_factor = model(coords, image)[1]
        mc_samples = mc_sample_cov_is_low_rank(mean_vector, low_rank_factor, number_of_samples, 4, False).to(device=DEVICE)
        for j in range(batch):
            for i in range(len(gts)):
                gt = gts[i][j].view(-1).to(device=DEVICE)
                for k in range(number_of_samples):
                    log_prob[j][i][k] = -loss_function(mc_samples[i][k][j], gt).to(device=DEVICE)
            losses[j] = torch.mean(-torch.logsumexp(log_prob[j].clone(), dim=1) + math.log(number_of_samples))
        print(losses)
    loss = torch.mean(losses)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("cross : ")
    #cross, gts_div, sample_div = gen_energy_distance(model, coords, 5, training_set, DEVICE, batch)
    #print("gts_div : ", gts_div)
    #print("sample_div : ", sample_div)
    print(epoch, loss)
    del image
    del mc_samples
    del gts
    del log_prob
    del losses

