import torch
from metrics import metrics
from torch.utils.tensorboard import SummaryWriter


def train(epoch, iterations_per_epoch, model, model_type, loss_function, optimizer, number_of_samples_per_gt,
          num_of_val_sample, coords, training_set, batch_size, device, writer):
    model.train()
    print("Train/epoch : ", epoch)
    comment = f' epoch = {epoch} model_type = {model_type} Training'
    tb = SummaryWriter(comment=comment)
    x=0
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
        if idx % 20 == 0:
            print(epoch, global_step)
            tb.add_scalar('Training/Loss', loss.item(), global_step=idx)
            tb.add_scalar('Training/Ged', ged.item(), global_step=idx)
            tb.add_scalar('Training/Cross', cross.item(), global_step=idx)
            tb.add_scalar('Training/Sample Diversity', sample_diversity.item(), global_step=idx)
            tb.add_scalar('Training/GTs Diversity', gts_diversity.item(), global_step=idx)
    tb.add_hparams(
        {"epoch": epoch, "model_type": model_type, "MC_samples": number_of_samples_per_gt,
         "MC_samples_GED": num_of_val_sample, 'batch_size' : batch_size},
        {
            "x": x,
        },
    )

    tb.close()
    
    
def validation(epoch, iterations_per_epoch, model, model_type, loss_function, number_of_samples_per_gt,
               num_of_val_sample, coords, validation_set, batch_size, device, writer):
    print("Validation/epoch : ", epoch)
    comment = f' epoch = {epoch} model_type = {model_type} Validation'
    tb = SummaryWriter(comment=comment)
    x=0
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
            tb.add_scalar('Training/Loss', loss.item(), global_step=idx)
            tb.add_scalar('Training/Ged', ged.item(), global_step=idx)
            tb.add_scalar('Training/Cross', cross.item(), global_step=idx)
            tb.add_scalar('Training/Sample Diversity', sample_diversity.item(), global_step=idx)
            tb.add_scalar('Training/GTs Diversity', gts_diversity.item(), global_step=idx)
        tb.add_hparams(
            {"epoch": epoch, "model_type": model_type, "MC_samples": number_of_samples_per_gt,
             "MC_samples_GED": num_of_val_sample, 'batch_size': batch_size},
            {
                "x": x,
            },
        )

        tb.close()
   
