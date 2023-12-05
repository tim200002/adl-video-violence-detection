def exponential_decay(optimizer, epoch, decay_rate=0.96, lr_decay_epoch=2):
    """Decay learning rate by a factor of decay_rate every lr_decay_epoch epochs."""
    if epoch == 0:
        return optimizer
    
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        print("Exponential Decay updating learning rate")
        param_group['lr'] *= decay_rate

    return optimizer

def exponential_decay_iter(optimizer, iteration, decay_rate, decay_every_iter):
    """Decay learning rate by a factor of decay_rate every lr_decay_epoch epochs."""
    assert 0<decay_rate<=1
    assert iteration>=0
    assert decay_every_iter>0

    if(decay_rate==1):
        return optimizer
    
    if iteration == 0:
        return optimizer
    
    if iteration % decay_every_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_rate

    return optimizer