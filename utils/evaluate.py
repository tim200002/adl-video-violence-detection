def evaluate(model, data_load, loss_history):
    model.eval()

    samples = len(data_load.dataset)
    accuracy = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data,  target in data_load:
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            accuracy += pred.eq(target.cuda()).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_history.append(aloss)
    print('\nAverage loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(accuracy) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * accuracy / samples) + '%)\n')
    
    return accuracy