from metrics import loss_func, metrics_val
import torch


def train(epoch, model, model_type, loss_function, optimizer, number_of_samples_per_gt,
          coords, training_set, batch_size, device, writer):
    model.train()
    print("Train/epoch : ", epoch)
    for idx, (image, gts, uid) in enumerate(training_set):
        loss = loss_func(image, gts, model, model_type, loss_function, number_of_samples_per_gt, coords, batch_size,
                         device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step = (epoch * len(training_set)) + idx
        if global_step % 20 == 0:
            print(epoch, global_step)
            writer.add_scalar('Training/Loss', loss.item(), global_step=global_step)

'''
def validation(epoch, iterations_per_epoch,iterations_per_epoch2, model, model_type, loss_function, number_of_samples_per_gt,
               num_of_val_sample, coords, validation_set, batch_size, device, writer):
    print("Validation/epoch : ", epoch)
    model.eval()
    total_ged = 0
    total_sample_div = 0
    total_gts_div = 0
    total_cross = 0
    total_dice = 0
    total_num_slices_used = 0
    total_dice_nod = 0
    total_num_slices_nod = 0
    total_loss = 0

    for idx, (image, gts, uid) in enumerate(validation_set):
        loss = loss_func(image, gts, model, model_type, loss_function, number_of_samples_per_gt, coords, batch_size,
                         device)
        ged, cross, gts_div, sample_div, dice_nod, num_slices_nod = metrics_val(image,gts, model,
                                                                                                       model_type,
                                                                                                       num_of_val_sample,
                                                                                                       coords, batch_size,
                                                                                                       device)
        total_ged += ged
        total_gts_div += gts_div
        total_sample_div += sample_div
        total_cross += cross
        #total_dice += dice
        #total_num_slices_used += num_slices_used
        total_dice_nod += dice_nod
        total_num_slices_nod += num_slices_nod
        #total_loss += loss
        del loss
        global_step = (epoch + 1 ) * iterations_per_epoch2

    total_ged = total_ged / iterations_per_epoch
    total_sample_div = total_sample_div / iterations_per_epoch
    total_gts_div = total_gts_div / iterations_per_epoch
    total_cross = total_cross / iterations_per_epoch
    total_dice = total_dice / total_num_slices_used
    total_dice_nod = total_dice_nod / total_num_slices_nod
    writer.add_scalar('Validation/Ged', total_ged.item(), global_step=global_step)
    writer.add_scalar('Validation/Sample Diversity', total_sample_div.item(), global_step=global_step)
    writer.add_scalar('Validation/GTs Diversity', total_gts_div.item(), global_step=global_step)
    writer.add_scalar('Validation/Cross', total_cross.item(), global_step=global_step)
    #writer.add_scalar('Validation/Dice', total_dice.item(), global_step=global_step)
    writer.add_scalar('Validation/Dice_Nod', total_dice_nod.item(), global_step=global_step)
    #writer.add_scalar('Validation/Loss', total_loss.item(), global_step=global_step)

'''
def validation(epoch, model, model_type, loss_function, number_of_samples_per_gt,
               num_of_val_sample, coords, validation_set, batch_size, device, writer, training_set):
    print("Validation/epoch : ", epoch)
    print(len(training_set))
    total_ged = 0
    total_sample_div = 0
    total_gts_div = 0
    total_cross = 0
    total_dice = 0
    total_num_slices_used = 0
    total_dice_nod = 0
    total_num_slices_nod = 0
    total_loss = 0
    with torch.no_grad():
        for idx, (image, gts, uid) in enumerate(validation_set):
            loss = loss_func(image, gts, model, model_type, loss_function, number_of_samples_per_gt, coords, batch_size,
                             device)

            ged, cross, gts_div, sample_div, dice_nod, num_slices_nod, dice, num_slices_used = metrics_val(image,gts, model,
                                                                                                           model_type,
                                                                                                           num_of_val_sample,
                                                                                                           coords, batch_size,
                                                                                                           device)
            total_ged += ged
            total_gts_div += gts_div
            total_sample_div += sample_div
            total_cross += cross
            total_dice += dice
            total_num_slices_used += num_slices_used
            total_dice_nod += dice_nod
            total_num_slices_nod += num_slices_nod
            total_loss += loss
        global_step = (epoch + 1) * len(training_set)
        total_ged = total_ged / len(validation_set)
        total_sample_div = total_sample_div / len(validation_set)
        total_gts_div = total_gts_div / len(validation_set)
        total_cross = total_cross / len(validation_set)
        total_dice = total_dice / total_num_slices_used
        total_dice_nod = total_dice_nod / total_num_slices_nod
        total_loss = total_loss / len(validation_set)
        writer.add_scalar('Validation/Ged', total_ged.item(), global_step=global_step)
        writer.add_scalar('Validation/Sample Diversity', total_sample_div.item(), global_step=global_step)
        writer.add_scalar('Validation/GTs Diversity', total_gts_div.item(), global_step=global_step)
        writer.add_scalar('Validation/Cross', total_cross.item(), global_step=global_step)
        writer.add_scalar('Validation/Dice', total_dice.item(), global_step=global_step)
        writer.add_scalar('Validation/Dice_Nod', total_dice_nod.item(), global_step=global_step)
        writer.add_scalar('Validation/Loss', total_loss.item(), global_step=global_step)



