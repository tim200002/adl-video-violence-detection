class InfinityDomainSampler:
    def __init__(self, dataloader):
        """
        creates a sampler which can just infinitly sample new data. It can be created in two ways:
        a) InfinityDomainSampler(dataloader)
        """
        self.dataloader = dataloader
        self.domain_dataloder_iter = enumerate(self.dataloader)

    def get_sample(self):
        batch, self.domain_dataloder_iter = get_next_element_from_dataloader_or_start_again(self.domain_dataloder_iter, self.dataloader)
        return batch
    


def get_next_element_from_dataloader_or_start_again(dataloader_iter, data_loader):
    try:
        # try to get next element
        _, batch = dataloader_iter.__next__()
    except StopIteration:
        # if the end is reached, start again
        dataloader_iter = enumerate(data_loader)
        _, batch = dataloader_iter.__next__()
    
    return batch, dataloader_iter
