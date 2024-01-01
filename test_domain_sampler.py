from utils.sampler import InfinityDomainSampler


if __name__ == "__main__":
    import configs.config_mmd as config
    data_loader_1 = config.valid_loader_ucf_small
    data_loader_2 = config.valid_loader_hockey

    sampler1 = InfinityDomainSampler(data_loader_1, config.Bs_Test)
    sampler2 = InfinityDomainSampler(data_loader_2, config.Bs_Test)

    while True:
        (videos1, labels1) = sampler1.get_sample()
        (videos2, labels2) = sampler2.get_sample()

        if videos1.shape != videos2.shape:
            print("Different shapes")
            print(videos1.shape)
            print(videos2.shape)
            break
        if labels1.shape != labels2.shape:
            print("Different shapes")
            print(labels1.shape)
            print(labels2.shape)
            break
