import torch
from metrics import metrics


def train(epoch, iterations_per_epoch, model, model_type, loss_function, optimizer, number_of_samples_per_gt,
          num_of_val_sample, coords, training_set, batch_size, device, writer):
    model.train()
    print("Train/epoch : ", epoch)
    for idx, (image, gts, uid) in enumerate(training_set):
        loss, ged, cross, gts_diversity, sample_diversity = metrics(image, gts, model, model_type, loss_function,
                                                                    number_of_samples_per_gt, num_of_val_sample,
                                                                    coords, batch_size, device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step = (epoch * iterations_per_epoch) + idx
        if global_step % 20 == 0:
            print(epoch, global_step)
            writer.add_scalar('Training/Loss', loss.item(), global_step=global_step)
            writer.add_scalar('Training/Ged', ged.item(), global_step=global_step)
            writer.add_scalar('Training/Cross', cross.item(), global_step=global_step)
            writer.add_scalar('Training/Sample Diversity', sample_diversity.item(), global_step=global_step)
            writer.add_scalar('Training/GTs Diversity', gts_diversity.item(), global_step=global_step)
    
    
def validation(epoch, iterations_per_epoch, model, model_type, loss_function, number_of_samples_per_gt,
               num_of_val_sample, coords, validation_set, batch_size, device, writer):
    print("Validation/epoch : ", epoch)
    with torch.no_grad():
        for idx, (image, gts, uid) in enumerate(validation_set):
            loss, ged, cross, gts_diversity, sample_diversity = metrics(image, gts, model, model_type, loss_function,
                                                                        number_of_samples_per_gt, num_of_val_sample,
                                                                        coords, batch_size, device)

            global_step = (epoch * iterations_per_epoch) + idx
            writer.add_scalar('Validation/Loss', loss.item(), global_step=global_step)
            writer.add_scalar('Validation/Ged', ged.item(), global_step=global_step)
            writer.add_scalar('Validation/Cross', cross.item(), global_step=global_step)
            writer.add_scalar('Validation/Sample Diversity', sample_diversity.item(), global_step=global_step)
            writer.add_scalar('Validation/GTs Diversity', gts_diversity.item(), global_step=global_step)
   
