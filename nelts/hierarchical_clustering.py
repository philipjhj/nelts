from scipy.cluster import hierarchy
from scipy.stats import zscore
import numpy as np

from matplotlib import pyplot as plt
from time import sleep

import ipdb


class HierarchicalClustering:
    """ Hierarchical clusering implemented to used with NELTS

        Steps:
            -


    """

    def __init__(self, w, max_subtree_size, n_top_subtrees, linkage_method='average'):
        self.linkage_method = linkage_method  # 'average' used in MATLAB implementation
        self.w = w
        self.max_subtree_size = max_subtree_size
        self.n_top_subtrees = n_top_subtrees

    def fit_offline(self, sample_sequence, sequence_length, n_samples):
        """ Computes the baseline mean and std for subtrees of size 2 to
        max_subtree_size

        Parameters:
            sample_sequence: 1-d sample sequence from the domain

            n_samples: number of samples used to compute baselines for each size
            of the subtrees.
        """

        n = sample_sequence.shape[1]-sequence_length
        subtree_dists = np.zeros((self.max_subtree_size-1, int(n_samples)))

        k = 0
        while not np.all(np.count_nonzero(subtree_dists, axis=1) == n_samples):
            k += 1
            if k % int(n_samples*0.05) == 0:
                #print(np.count_nonzero(subtree_dists, axis=1))
                pass
            subsequences = np.empty((self.w, sequence_length))

            for i in range(self.w):
                idx = np.random.randint(n)
                subsequence = sample_sequence[0, idx:idx+sequence_length]

                subsequence_preprocessed = self.preprocess(subsequence)

                if i % 2 == 0:
                    #subsequence_flipped = np.fliplr(subsequence_preprocessed)
                    subsequence_flipped = subsequence_preprocessed[::-1]
                else:
                    subsequence_flipped = 1-subsequence_preprocessed

                subsequences[i, :] = subsequence_flipped

            Z = hierarchy.linkage(subsequences,
                                  method=self.linkage_method)

            full_tree = hierarchy.to_tree(Z)

            subtrees = self.find_subtrees(full_tree)

            info_subtrees = self.extract_subtree_info(subtrees,
                                                      compute_score=False)

            for info_subtree in info_subtrees:
                size_idx = info_subtree['size']-2

                # Check if desired amount samples have been found for current
                # subtree size
                dists = subtree_dists[size_idx, :]
                if np.count_nonzero(dists) < n_samples:
                    next_place_idx = np.where(dists == 0)[0][0]

                    subtree_dists[size_idx,
                                  next_place_idx] = info_subtree['dist']
                else:
                    continue

        self.baseline_mean = np.mean(subtree_dists, axis=1)
        self.baseline_std = np.std(subtree_dists, axis=1)

    @staticmethod
    def preprocess(X):
        # Normalize
        X_processed = zscore(X, ddof=1)
        return X_processed

    def most_significant_subtree(self, X):
        """ X assumed to be normalized already

        """
        self.Z = hierarchy.linkage(X, method=self.linkage_method)
        full_tree = hierarchy.to_tree(self.Z)

        subtrees = self.find_subtrees(full_tree)
        nonoverlapping_subtrees = self.extract_nonoverlapping_subtrees(
            subtrees)

        self.top_leafs = [leaf for subtree in
                          nonoverlapping_subtrees[:self.n_top_subtrees]
                          for leaf in subtree['leafs']]

        top_tree = nonoverlapping_subtrees[0]

        return top_tree

    def find_subtrees(self, tree, subtrees=[]):
        """ Extracts all subtrees of size 2 to max_subtree_size

            TODO: make this more efficient by setting a minimum size limit
        """

        try:
            subtrees = self.find_subtrees(tree.get_left(), subtrees)
            subtrees = self.find_subtrees(tree.get_right(), subtrees)
        except AttributeError:
            # a leaf has been reached
            pass

        if 1 < tree.get_count() < self.max_subtree_size+1:
            subtrees = subtrees + [tree]

        return subtrees

    def extract_nonoverlapping_subtrees(self, subtrees):
        """ Extracts subtrees with no overlap in order of significance. I.e.
        leaf nodes will only be contained in their most significant subtree as
        defined by the score
        """
        info_subtrees = self.extract_subtree_info(subtrees, compute_score=True)

        self.nonoverlapping_subtrees = []
        seen_leafs = set()

        for info_subtree in info_subtrees:
            leafs = info_subtree['leafs']
            if np.any(np.isin(leafs, list(seen_leafs))):
                seen_leafs = seen_leafs.union(set(leafs))
                continue

            self.nonoverlapping_subtrees.append(info_subtree)

            seen_leafs = seen_leafs.union(set(leafs))

        return self.nonoverlapping_subtrees

    def extract_subtree_info(self, subtrees, compute_score=False):
        info_subtrees = []

        for subtree in subtrees:
            info_subtree = {}
            info_subtree['leafs'] = subtree.pre_order()
            info_subtree['size'] = subtree.get_count()
            info_subtree['dist'] = subtree.dist
            if compute_score:
                info_subtree['score'] = self.compute_subtree_score(
                    info_subtree['size'],
                    info_subtree['dist'])

            info_subtrees.append(info_subtree)

        if compute_score:
            sort_key = 'score'
        else:
            sort_key = 'dist'

        info_subtrees = sorted(
            info_subtrees, key=lambda k: k[sort_key], reverse=True)

        return info_subtrees

    def compute_subtree_score(self, subtree_size, subtree_dist):
        """ Compute score for a subtree
        """

        score = (self.baseline_mean[subtree_size-2] -
                 subtree_dist)/self.baseline_std[subtree_size-2]

        return score
