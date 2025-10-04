from Unet_Architecture.Image_Painting.Library import *

#perceptual_loss_rate = 0.05 # bạn muốn dùng 0.05 cho perceptual loss

def generate_images_from_batch(inputs, predictions, labels, mask_learn=None, max_display=1):
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    if mask_learn is not None:
        mask_learn = mask_learn.cpu().numpy()

    batch_size = inputs.shape[0]
    num_show = min(max_display, batch_size)

    for i in range(batch_size - num_show, batch_size):
        plt.figure(figsize=(16, 4))
        titles = ['Input (masked)', 'Prediction', 'Ground Truth']
        images = [inputs[i], predictions[i], labels[i]]

        if mask_learn is not None:
            titles.append('Mask Learn')
            images.append(mask_learn[i])

        for j, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, len(images), j + 1)
            img_show = img.squeeze()
            plt.imshow(img_show, cmap='gray')
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


def plot_result(num_epochs, train_psnrs, eval_psnrs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(epochs, train_psnrs, label="Training")
    axs[0].plot(epochs, eval_psnrs, label="Evaluation")
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("PSNR")
    axs[1].set_ylabel("Loss")
    plt.legend()


def predict_and_display(model, test_dataloader, device, max_batches=10):
    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels, mask_learn) in enumerate(test_dataloader):
            if idx >= max_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask_learn = mask_learn.to(device)

            predictions = torch.sigmoid(model(inputs))
            generate_images_from_batch(inputs, predictions, labels, mask_learn, max_display=10)

def train_epoch(model, optimizer, criterion_bce, criterion_lpips,
                train_loader, device, epoch,
                perceptual_loss_rate, log_interval=100):

    model.train()
    total_psnr, total_count = 0, 0
    losses = []
    
    for idx, (inputs, labels, mask_learn) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # raw logits

        # Áp dụng sigmoid trước khi tính loss
        outputs_sigmoid = torch.sigmoid(outputs)

        # Loss toàn ảnh
        loss_bce = criterion_bce(outputs_sigmoid, labels)

        if perceptual_loss_rate > 0:
            loss_lpips = criterion_lpips(outputs_sigmoid, labels)
            loss = loss_bce + perceptual_loss_rate * loss_lpips.mean()
        else:
            loss = loss_bce

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # Dice coefficent
        outputs_sigmoid = torch.sigmoid(outputs)
        dice_score = dice_coefficient(outputs_sigmoid, labels)
        # PSNR toàn ảnh
        #total_psnr += peak_signal_noise_ratio(outputs_sigmoid, labels)
        total_psnr += peak_signal_noise_ratio(outputs_sigmoid, labels,data_range=1.0)
        total_count += 1

        if idx % log_interval == 0 and idx > 0:
            avg_psnr = total_psnr / total_count
            print(f"[Train] Epoch {epoch:3d} | Batch {idx:5d}/{len(train_loader)} | Avg PSNR: {avg_psnr:.2f}")
            total_psnr, total_count = 0, 0

    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_loss

def valid_epoch(model, criterion_bce, criterion_lpips,
                val_loader, device, perceptual_loss_rate, show_debug=False):
    dice_scores = []
    model.eval()
    total_psnr, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels, mask_learn) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs_sigmoid = torch.sigmoid(outputs)

            # -------- DEBUG: kiểm tra dữ liệu --------
            if idx == 0 and show_debug:
                print(f"[DEBUG] Input min/max: {inputs.min().item():.3f} / {inputs.max().item():.3f}")
                print(f"[DEBUG] Label min/max: {labels.min().item():.3f} / {labels.max().item():.3f}")
                print(f"[DEBUG] Output sigmoid min/max: {outputs_sigmoid.min().item():.3f} / {outputs_sigmoid.max().item():.3f}")

                for i in range(min(2, inputs.size(0))):
                    inp = inputs[i].cpu().numpy().squeeze()
                    out = outputs_sigmoid[i].cpu().numpy().squeeze()
                    tgt = labels[i].cpu().numpy().squeeze()

                    plt.figure(figsize=(12, 3))
                    plt.subplot(1, 3, 1)
                    plt.title("Input")
                    plt.imshow(inp, cmap='gray')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.title("Prediction")
                    plt.imshow(out, cmap='gray')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.title("Ground Truth")
                    plt.imshow(tgt, cmap='gray')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
            # ------------------------------------------

            # Tính loss
            loss_bce = criterion_bce(outputs_sigmoid, labels)
            if perceptual_loss_rate > 0:
                loss_lpips = criterion_lpips(outputs_sigmoid, labels)
                loss = loss_bce + perceptual_loss_rate * loss_lpips.mean()
            else:
                loss = loss_bce

            losses.append(loss.item())
            total_psnr += peak_signal_noise_ratio(outputs_sigmoid, labels, data_range=1.0)
            total_count += 1

            # Tính Dice
            dice = dice_coefficient(outputs_sigmoid, labels)
            dice_scores.append(dice.item())

    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0

    print(f"[Validation] Dice: {avg_dice:.4f} | Loss: {epoch_loss:.4f} | PSNR: {epoch_psnr:.2f} dB")
    return epoch_psnr, epoch_loss, avg_dice


def training_1pos_2crit(model, optimizer, criterion_bce, criterion_lpips, train_loader, val_loader, num_epochs, device, perceptual_loss_rate):
    train_psnrs, train_losses = [], []
    eval_psnrs, eval_losses = [], []
    best_dice = 0.0

    for epoch in range(1, num_epochs + 1):
        time_start = time.time()

        #Train in 1 epoch
        train_psnr, train_loss = train_epoch(
            model, optimizer, criterion_bce, criterion_lpips, train_loader, device, epoch,
            perceptual_loss_rate
        )
        val_psnr, val_loss,_ = valid_epoch(
            model, criterion_bce, criterion_lpips, val_loader, device,
            perceptual_loss_rate
        )

        #Save Result
        train_psnrs.append(train_psnr.cpu())
        train_losses.append(train_loss)
        eval_psnrs.append(val_psnr.cpu())
        eval_losses.append(val_loss)
        # Hiển thị ảnh sau mỗi 50 epoch
        if epoch % 50 == 0:
            inputs, labels, mask_learn = next(iter(val_loader))
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask_learn = mask_learn.to(device)
            with torch.no_grad():
                predictions = torch.sigmoid(model(inputs))
            generate_images_from_batch(inputs, predictions, labels, mask_learn, max_display=10)
        print("-" * 60)
        print(
            f"| End of epoch: {epoch:3d} | Time: {time.time() - time_start:.2f}s | "
            f"Train PSNR: {train_psnr:.3f} | Train Loss: {train_loss:.3f} | "
            f"Val PSNR: {val_psnr:.3f} | Val Loss: {val_loss:.3f} "
        )
        print("-" * 60)

    model.eval()
    metrics = {
        "train_psnr": train_psnrs,
        "train_losses": train_losses,
        "val_psnr": eval_psnrs,
        "val_losses": eval_losses
    }
    return model, metrics



def training_mulPos_2crit(model, optimizer, criterion_bce, criterion_lpips, train_loader, val_loader, max_epochs, max_random_round, device, perceptual_loss_rate):
    train_psnrs, train_losses = [], []
    eval_psnrs, eval_losses = [], []
    best_dice = 0.0

    for epoch in range(1, max_epochs + 1):
        time_start = time.time()

        #Train in 1 epoch
        train_psnr, train_loss = train_epoch(
            model, optimizer, criterion_bce, criterion_lpips, train_loader, device, epoch,
            perceptual_loss_rate
        )
        val_psnr, val_loss,_ = valid_epoch(
            model, criterion_bce, criterion_lpips, val_loader, device,
            perceptual_loss_rate
        )

        #Save Result
        train_psnrs.append(train_psnr.cpu())
        train_losses.append(train_loss)
        eval_psnrs.append(val_psnr.cpu())
        eval_losses.append(val_loss)
        # Hiển thị ảnh sau mỗi 10 epoch
        if epoch % 20 == 0:
            inputs, labels, mask_learn = next(iter(val_loader))
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask_learn = mask_learn.to(device)
            with torch.no_grad():
                predictions = torch.sigmoid(model(inputs))
            generate_images_from_batch(inputs, predictions, labels, mask_learn, max_display=2)
        print("-" * 60)
        print(
            f"| End of epoch: {epoch:3d} | Time: {time.time() - time_start:.2f}s | "
            f"Train PSNR: {train_psnr:.3f} | Train Loss: {train_loss:.3f} | "
            f"Val PSNR: {val_psnr:.3f} | Val Loss: {val_loss:.3f} "
        )
        print("-" * 60)

    model.eval()
    metrics = {
        "train_psnr": train_psnrs,
        "train_losses": train_losses,
        "val_psnr": eval_psnrs,
        "val_losses": eval_losses
    }
    return model, metrics



'''
def train_epoch_May18(model, optimizer, criterion_bce, criterion_lpips, train_loader, device, epoch, perceptual_loss_rate = perceptual_loss_rate, log_interval=100):
    model.train()
    total_psnr, total_count = 0, 0
    losses = []

    for idx, (inputs, labels, mask_learn) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        mask_learn = mask_learn.to(device)

        optimizer.zero_grad()
        # outputs: raw logits
        outputs = model(inputs)

        # tính loss toàn ảnh
        loss_bce = criterion_bce(outputs, labels)

        if perceptual_loss_rate > 0:
            loss_lpips = criterion_lpips(torch.sigmoid(outputs), labels)
            loss = loss_bce + perceptual_loss_rate * loss_lpips.mean()
        else:
            loss = loss_bce

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # PSNR toàn ảnh (sau sigmoid)
        outputs_sigmoid = torch.sigmoid(outputs)
        total_psnr += peak_signal_noise_ratio(outputs_sigmoid, labels)
        total_count += 1

        if idx % log_interval == 0 and idx > 0:
            avg_psnr = total_psnr / total_count
            print(f"[Train] Epoch {epoch:3d} | Batch {idx:5d}/{len(train_loader)} | Avg PSNR: {avg_psnr:.2f}")
            total_psnr, total_count = 0, 0

    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_loss

def valid_epoch_May18(model, criterion_bce, criterion_lpips, val_loader, device, perceptual_loss_rate=0.0):
    model.eval()
    total_psnr, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels, mask_learn) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask_learn = mask_learn.to(device)

            outputs = model(inputs)
            loss_bce = criterion_bce(outputs, labels)

            if perceptual_loss_rate > 0:
                loss_lpips = criterion_lpips(torch.sigmoid(outputs), labels)
                loss = loss_bce + perceptual_loss_rate * loss_lpips.mean()
            else:
                loss = loss_bce

            losses.append(loss.item())

            predictions = torch.sigmoid(outputs)
            total_psnr += peak_signal_noise_ratio(predictions, labels)
            total_count += 1

    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)

    print(f"[Validation] Loss: {epoch_loss:.4f} | PSNR: {epoch_psnr:.2f} dB")
    return epoch_psnr, epoch_loss
'''