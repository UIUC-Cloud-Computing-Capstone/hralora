"""
Non-IID data partitioners for federated learning.

Splits datasets by class (and optionally by domain) across users with configurable
distribution (Dirichlet or uniform). Reference: https://github.com/taokz/FeDepth
"""
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import copy
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter

class _Sampler(object):
    """Base sampler: holds a deep copy of an array; next() is implemented by subclasses."""

    def __init__(self, arr):
        self.arr = copy.deepcopy(arr)

    def next(self):
        raise NotImplementedError()


class shuffle_sampler(_Sampler):
    """
    Sampler that yields elements from a shuffled array one-by-one; reshuffles when exhausted.
    """

    def __init__(self, arr, rng=None):
        super().__init__(arr)
        if rng is None:
            rng = np.random
        rng.shuffle(self.arr)
        self._idx = 0
        self._max_idx = len(self.arr)

    def next(self):
        if self._idx >= self._max_idx:
            np.random.shuffle(self.arr)
            self._idx = 0
        v = self.arr[self._idx]
        self._idx += 1
        return v

class Partitioner(object):
    """Partition a total sample count into multiple shares (users) via Dirichlet or uniform split.

    Each share gets at least min_n_sample_per_share; the remainder is split according to
    partition_mode. Dirichlet uses dir_par_beta as the concentration parameter.

    Args:
        rng (np.random.RandomState or None): Random state; default np.random.
        partition_mode (str): 'dir' for Dirichlet, 'uni' for equal split.
        max_n_sample_per_share (int): Cap samples per share (-1 = no cap).
        min_n_sample_per_share (int): Minimum samples per share (default 2).
        max_n_sample (int): Cap total samples (-1 = no cap).
        verbose (bool): If True, log partition info via log().
        dir_par_beta (float): Dirichlet concentration (default 1).
    """
    def __init__(self, rng=None, partition_mode="dir",
                 max_n_sample_per_share=-1,
                 min_n_sample_per_share=2,
                 max_n_sample=-1,
                 verbose=True,
                 dir_par_beta=1
                 ):
        assert max_n_sample_per_share < 0 or max_n_sample_per_share > min_n_sample_per_share, \
            f"max ({max_n_sample_per_share}) > min ({min_n_sample_per_share})"
        self.rng = rng if rng else np.random
        self.partition_mode = partition_mode
        self.max_n_sample_per_share = max_n_sample_per_share
        self.min_n_sample_per_share = min_n_sample_per_share
        self.max_n_sample = max_n_sample
        self.verbose = verbose
        self.dir_par_beta = dir_par_beta

    def __call__(self, n_sample, n_share, log=print):
        """Split n_sample into n_share shares with min/max and distribution constraints.

        Args:
            n_sample (int): Total number of samples to partition.
            n_share (int): Number of shares (users).
            log (callable): Logging function (default print).

        Returns:
            list: Length n_share; partition[i] is the number of samples for share i.
                  Sum equals n_sample; each element >= min_n_sample_per_share.
        """
        assert n_share > 0, f"cannot split into {n_share} share"
        if self.verbose:
            log(f"  {n_sample} smp => {n_share} shards by {self.partition_mode} distribution")
        if self.max_n_sample > 0:
            n_sample = min((n_sample, self.max_n_sample))
        if self.max_n_sample_per_share > 0:
            n_sample = min((n_sample, n_share * self.max_n_sample_per_share))

        if n_sample < self.min_n_sample_per_share * n_share:
            raise ValueError(f"Not enough samples. Require {self.min_n_sample_per_share} samples"
                             f" per share at least for {n_share} shares. But only {n_sample} is"
                             f" available totally.")
        n_sample -= self.min_n_sample_per_share * n_share
        if self.partition_mode == "dir":
            partition = (self.rng.dirichlet(n_share * [self.dir_par_beta]) * n_sample).astype(int)
        elif self.partition_mode == "uni":
            partition = int(n_sample // n_share) * np.ones(n_share, dtype='int')
        else:
            raise ValueError(f"Invalid partition_mode: {self.partition_mode}")

        # uniformly add residual to as many users as possible.
        for i in self.rng.choice(n_share, n_sample - np.sum(partition)):
            partition[i] += 1
            # partition[-1] += n_sample - np.sum(partition)  # add residual
        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        partition = partition + self.min_n_sample_per_share
        n_sample += self.min_n_sample_per_share * n_share
        # partition = np.minimum(partition, max_n_sample_per_share)
        partition = partition.tolist()

        assert sum(partition) == n_sample, f"{sum(partition)} != {n_sample}"
        assert len(partition) == n_share, f"{len(partition)} != {n_share}"
        return partition

class ClassWisePartitioner(Partitioner):
    """Partition labeled data across users by class (and optionally by domain).

    Each user is assigned n_class_per_share classes (shuffled); within each class,
    samples are split across the users that have that class using the parent
    Partitioner (Dirichlet or uniform). If domains is provided, splitting within
    a class is done per domain first, then assigned to users.

    Args:
        n_class_per_share (int): Number of classes per user (default 2).
        dir_par_beta (float): Passed to Partitioner for within-class split (default 1).
        n_domain_per_share (int): Used when domains is None; must be > 1 (default 6).
        **kwargs: Passed to Partitioner (rng, partition_mode, min/max_n_sample*, verbose).
    """
    def __init__(self, n_class_per_share=2, dir_par_beta=1, n_domain_per_share=6, **kwargs):
        super(ClassWisePartitioner, self).__init__(**kwargs)
        self.n_class_per_share = n_class_per_share
        self.dir_par_beta = dir_par_beta
        self.n_domain_per_share = n_domain_per_share
        self._aux_partitioner = Partitioner(dir_par_beta=dir_par_beta,**kwargs)

    def __call__(self, labels, n_user, log=print, user_ids_by_class=None,
                 return_user_ids_by_class=False, consistent_class=False, domains=None):
        """Partition sample indices by class (and optionally domain) across n_user users.

        Labels are grouped by class; each user gets n_class_per_share classes (via
        shuffle_sampler). Within each class, samples are split among the users that
        have that class (Dirichlet/uni via _aux_partitioner). If domains is provided,
        within-class split is done by domain first, then assigned to users.

        Args:
            labels: Sequence or 1D tensor of label per sample (length = dataset size).
            n_user (int): Number of users (shares).
            log (callable): Logging function (default print).
            user_ids_by_class (dict, optional): Precomputed class -> list of user ids; if None, built from shuffle.
            return_user_ids_by_class (bool): If True, also return user_ids_by_class (default False).
            consistent_class (bool): If True, use self.rng for class shuffle for reproducibility.
            domains (sequence, optional): Domain id per sample; if provided, enables domain-aware split.

        Returns:
            If return_user_ids_by_class is False: list of length n_user; idx_by_user[i] is list of sample indices for user i.
            If True: (idx_by_user, user_ids_by_class) where user_ids_by_class maps class_id -> list of user ids.
        """
        if domains:
            # reorganize labels by class
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            idx_by_class = defaultdict(list)
            idx_by_domain = defaultdict(list)
            if len(labels) > 1e5:
                labels_iter = tqdm(labels, leave=False, desc='sort labels')
            else:
                labels_iter = labels
                domain_iter = domains
            for i, label in enumerate(labels_iter):
                idx_by_class[label].append(i)
            if domains:
                for i, domain in enumerate(domain_iter):
                    idx_by_domain[domain].append(i)

            n_class = len(idx_by_class)
            if domains:
                n_domain = len(idx_by_domain)
            # assert n_user * self.n_class_per_share > n_class, f"Cannot split {n_class} classes into " \
                                                            #   f"{n_user} users when each user only " \
                                                            #   f"has {self.n_class_per_share} classes."

            # assign classes to each user.
            if user_ids_by_class is None:
                user_ids_by_class = defaultdict(list)
                label_sampler = shuffle_sampler(list(range(n_class)),
                                                self.rng if consistent_class else None)
                for s in range(n_user):
                    s_classes = [label_sampler.next() for _ in range(self.n_class_per_share)]
                    for c in s_classes:
                        user_ids_by_class[c].append(s)

            # assign sample indexes to clients
            idx_by_user = [[] for _ in range(n_user)]
            if n_class > 100 or len(labels) > 1e5:
                idx_by_class_iter = tqdm(idx_by_class, leave=True, desc='split cls')
                log = lambda log_s: idx_by_class_iter.set_postfix_str(log_s[:10])  # tqdm.write
            else:
                idx_by_class_iter = idx_by_class
            for c in idx_by_class_iter:
                l = len(idx_by_class[c])
                log(f" class-{c} => {len(user_ids_by_class[c])} shares")
                initial_domains = np.arange(n_domain)
                np.random.shuffle(initial_domains)
                extra_assignments = np.random.choice(n_domain, len(user_ids_by_class[c]) - n_domain, replace=True)
                domain_assignments = np.concatenate((initial_domains, extra_assignments))
                counter_domain_samples = Counter(domain_assignments)
                l_per_domain = l / n_domain
                in_class_domain_ids = np.array(domain_iter)[idx_by_class[c]]
                in_class_domain_id_dict = {}
                for i in range(n_domain):
                    in_class_domain_id_dict[i] = np.array(idx_by_class[c])[np.where(in_class_domain_ids == i)[0]]
                
                for i_user, i_domain in zip(user_ids_by_class[c], domain_assignments):
                    domain_sample_dividen = int(l_per_domain / counter_domain_samples[i_domain])
                    selected_ids = np.random.choice(in_class_domain_id_dict[i_domain],
                                                    size=domain_sample_dividen if domain_sample_dividen <= len(in_class_domain_id_dict[i_domain]) else len(in_class_domain_id_dict[i_domain]),
                                                    replace=False)
                    idx_by_user[i_user].extend(selected_ids)
                    in_class_domain_id_dict[i_domain] = np.array(list(set(in_class_domain_id_dict[i_domain]) - set(selected_ids)))

            if return_user_ids_by_class:
                return idx_by_user, user_ids_by_class
            else:
                return idx_by_user
        else:
            # reorganize labels by class
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            idx_by_class = defaultdict(list)
            if len(labels) > 1e5:
                labels_iter = tqdm(labels, leave=False, desc='sort labels')
            else:
                labels_iter = labels
            for i, label in enumerate(labels_iter):
                idx_by_class[label].append(i)

            n_class = len(idx_by_class)
            assert self.n_domain_per_share > 1, "Only support 1 domain per share"
            # assert n_user * self.n_class_per_share > n_class, f"Cannot split {n_class} classes into " \
                                                            #   f"{n_user} users when each user only " \
                                                            #   f"has {self.n_class_per_share} classes."

            # assign classes to each user.
            if user_ids_by_class is None:
                user_ids_by_class = defaultdict(list)
                label_sampler = shuffle_sampler(list(range(n_class)),
                                                self.rng if consistent_class else None)
                for s in range(n_user):
                    s_classes = [label_sampler.next() for _ in range(self.n_class_per_share)]
                    for c in s_classes:
                        # group the clients with the class type as the key
                        user_ids_by_class[c].append(s)

            # assign sample indexes to clients
            idx_by_user = [[] for _ in range(n_user)]
            if n_class > 100 or len(labels) > 1e5:
                idx_by_class_iter = tqdm(idx_by_class, leave=True, desc='split cls')
                log = lambda log_s: idx_by_class_iter.set_postfix_str(log_s[:10])  # tqdm.write
            else:
                idx_by_class_iter = idx_by_class
            for c in idx_by_class_iter:
                l = len(idx_by_class[c])
                log(f" class-{c} => {len(user_ids_by_class[c])} shares")
                l_by_user = self._aux_partitioner(l, len(user_ids_by_class[c]), log=log) # num of sample for each client id in this categories
                # partition each class samples into differnt users.
                base_idx = 0
                for i_user, tl in zip(user_ids_by_class[c], l_by_user): # i_user: client_id, tl: num of sample
                    idx_by_user[i_user].extend(idx_by_class[c][base_idx:base_idx+tl])
                    base_idx += tl
            if return_user_ids_by_class:
                return idx_by_user, user_ids_by_class
            else:
                return idx_by_user