from dataloaders.datasets import lits, pairwise_lits, chaos, pairwise_chaos
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    if args.dataset == 'lits':
        train_set = lits.LitsDataloader(args, data_phase='train', margin=5)
        val_set = lits.LitsDataloader(args, data_phase='val', margin=5)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'pairwise_lits':
        train_set = pairwise_lits.LitsDataloader(args, data_phase='train', margin=5)
        val_set = pairwise_lits.LitsDataloader(args, data_phase='val', margin=5)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'chaos':
        train_set = chaos.ChaosDataloader(args, data_phase='train', margin=5)
        val_set = chaos.ChaosDataloader(args, data_phase='val', margin=5)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == 'pairwise_chaos':
        train_set = pairwise_chaos.ChaosDataloader(args, data_phase='train', margin=5)
        val_set = pairwise_chaos.ChaosDataloader(args, data_phase='val', margin=5)
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        test_loader = None
        return train_loader, val_loader, test_loader, num_class
    else:
        raise NotImplementedError

