class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset in ['lits', 'pairwise_lits']:
            return '../data/minor_lits/'                            
        elif dataset in ['chaos', 'pairwise_chaos']:
            return '../data/cross_chaos_5/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
