import torch


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        (
            feat,
            out0,
        ) = net(input1, input2)

        loss_id = criterion_id(out0, labels)
        loss_tri, batch_acc = criterion_tri(feat, labels)
        correct += batch_acc / 2
        _, predicted = out0.max(1)
        correct += predicted.eq(labels).sum().item() / 2

        loss = loss_id + loss_tri
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print(
                "Epoch: [{}][{}/{}] "
                "Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "lr:{:.3f} "
                "Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) "
                "iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) "
                "TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) "
                "Accu: {:.2f}".format(
                    epoch,
                    batch_idx,
                    len(trainloader),
                    current_lr,
                    100.0 * correct / total,
                    batch_time=batch_time,
                    train_loss=train_loss,
                    id_loss=id_loss,
                    tri_loss=tri_loss,
                )
            )

    writer.add_scalar("total_loss", train_loss.avg, epoch)
    writer.add_scalar("id_loss", id_loss.avg, epoch)
    writer.add_scalar("tri_loss", tri_loss.avg, epoch)
    writer.add_scalar("lr", current_lr, epoch)
