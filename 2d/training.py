from metrics import loss_func, metrics_val


def train(epoch, iterations_per_epoch, model, model_type, loss_function, optimizer, number_of_samples_per_gt,
          coords, training_set, batch_size, device, writer):
    model.train()
    print("Train/epoch : ", epoch)
    for idx, (image, gts, uid) in enumerate(training_set):
        loss = loss_func(image, gts, model, model_type, loss_function, number_of_samples_per_gt, coords, batch_size,
                         device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step = (epoch * iterations_per_epoch) + idx
        if global_step % 20 == 0:
            print(epoch, global_step)
            writer.add_scalar('Training/Loss', loss.item(), global_step=global_step)


def validation(epoch, iterations_per_epoch, model, model_type, loss_function, number_of_samples_per_gt,
               num_of_val_sample, coords, validation_set, batch_size, device, writer):
    print("Validation/epoch : ", epoch)
    model.eval()
    total_ged = 0
    total_sample_div = 0
    total_gts_div = 0
    total_cross = 0
    total_dice = 0
    for idx, (image, gts, uid) in enumerate(validation_set):
        loss = loss_func(image, gts, model, model_type, loss_function, number_of_samples_per_gt, coords, batch_size,
                         device)
        ged, cross, gts_diversity, sample_diversity, dice = metrics_val(image, gts, model, model_type,
                                                                        num_of_val_sample, coords, batch_size, device)
        total_ged += ged
        total_gts_div += gts_diversity
        total_sample_div += sample_diversity
        total_cross += cross
        total_dice += dice
        global_step = (epoch * iterations_per_epoch) + idx
        writer.add_scalar('Validation/Loss', loss.item(), global_step=global_step)
    total_ged = total_ged / iterations_per_epoch
    total_sample_div = total_sample_div / iterations_per_epoch
    total_gts_div = total_gts_div / iterations_per_epoch
    total_cross = total_cross / iterations_per_epoch
    total_dice = total_dice / iterations_per_epoch

    print(total_dice)

    writer.add_scalar('Validation/Ged', total_ged.item(), global_step=epoch)
    writer.add_scalar('Validation/Sample Diversity', total_sample_div.item(), global_step=epoch)
    writer.add_scalar('Validation/GTs Diversity', total_gts_div.item(), global_step=epoch)
    writer.add_scalar('Validation/Cross', total_cross.item(), global_step=epoch)
    writer.add_scalar('Validation/Dice', total_dice.item(), global_step=epoch)
