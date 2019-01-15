from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from scipy.stats import zscore, norm
import numpy as np
import attr
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec

from . hierarchical_clustering import HierarchicalClustering

from .base import NeverEndingFramework, NeverEndingConceptLearner, Concept, \
    DataStreamPreprocessor, DataStream


from pdb import set_trace


@attr.s
class SequenceStream(DataStream):
    sequence = attr.ib()
    subsequence_length = attr.ib()
    dist_between_subsequences_threshold = attr.ib()
    labels = attr.ib()
    current_idx = attr.ib(default=0)

    def stream(self):
        if np.ndim(self.sequence) == 2:
            self.sequence = self.sequence.reshape(1, -1)

        current_subsequence = np.repeat(
            [float('inf')], self.subsequence_length)

        for idx in range(self.sequence.shape[1]-self.subsequence_length):
            next_candidate_subsequence = zscore(
                self.sequence[0, idx:idx+self.subsequence_length], ddof=1)

            euc_dist = np.linalg.norm(
                current_subsequence-next_candidate_subsequence)

            if euc_dist > self.dist_between_subsequences_threshold:
                self.current_idx = idx
                current_subsequence = next_candidate_subsequence.reshape(1, -1)
                label = self.get_label(idx)
                yield current_subsequence, label

    def get_label(self, idx):
        """ A simple function getting labels from a given list as in the example
        of the MATLAB implementation.
        """
        # print('providing label')
        return self.labels[idx]


class IdentityPreprocessor(DataStreamPreprocessor):

    def __call__(self, data_point):
        return data_point


# @attr.s
# class incrementalBHCConceptLearner(NeverEndingConceptLearner):
#    """Concept learner NELTS (Towards Never-Ending Learning for Time-Series)
#
#    """
#    cluster_buffer = attr.ib()
#    concepts = attr.ib(default={'labels': [], 'thresholds': [], 'counts': [
#    ], 'prototypes': []})  # Concepts are clusters with a label
#
#    def fit(self, data_point, label):
#
#    def is_known(self, data_point):
#
#    def plot_clusters(self):
#        if self.was_known:
#            return False
#
#        HC = self.frequent_pattern_maintenance.hierarchical_clustering
#        success = False
#        try:
#            top = HC.most_significant_subtree(self.cluster_buffer)
#
#            gs0 = gridspec.GridSpec(1, 2)
#            gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0])
#            gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1])
#            gs010 = gridspec.GridSpecFromSubplotSpec(
#                10, 1, subplot_spec=gs01[0])
#            gs011 = gridspec.GridSpecFromSubplotSpec(
#                1, 1, subplot_spec=gs01[1])
#            # gs02 = gridspec.GridSpecFromSubplotSpec(20, 5, subplot_spec=gs0[2])
#
#            gs001 = gridspec.GridSpecFromSubplotSpec(
#                self.cluster_buffer.shape[0], 1, subplot_spec=gs00[1], wspace=0.35)
#
#            fig = plt.figure(figsize=(15, 15))
#            ax_dendogram = fig.add_subplot(gs00[0])
#            dn = hierarchy.dendrogram(
#                HC.Z, orientation='left', get_leaves=True, ax=ax_dendogram)
#
#            ax_leafs = []
#            steps = int(20/(self.cluster_buffer.shape[0]))
#            for i in range(self.cluster_buffer.shape[0]):
#                ax_leafs.append(fig.add_subplot(
#                    gs001[i, :]))
#
#            for i, lidx in enumerate(dn['leaves']):
#                ax_leafs[i].plot(self.cluster_buffer[lidx, :])
#                ax_leafs[i].axis('off')
#
#            ax_clusters = []
#            for i, subtree in enumerate(HC.nonoverlapping_subtrees):
#                ax_clusters.append(fig.add_subplot(gs010[i:(i+1), :]))
#                for leaf in subtree['leafs']:
#                    ax_clusters[i].plot(
#                        self.cluster_buffer[leaf, :], 'b')
#
#                mean_pattern = np.mean(
#                    self.cluster_buffer[subtree['leafs'], :], axis=0)
#
#                ax_clusters[i].plot(mean_pattern, 'r', linewidth=5)
#
#            try:
#                self.plot_concepts(gs011)
#            except ValueError:
#                pass
#            success = True
#
#        except ValueError:
#            pass
#
#        return success
#
#    def plot_concepts(self, gridspec):
#        pca = PCA(n_components=2)
#        prototypes = np.vstack(self.concepts['prototypes'])
#        prototypes_2d = pca.fit_transform(prototypes)
#
#        colors = 'rgby'  # TODO: make this dynamic
#
#        fig = plt.gcf()
#        ax1 = fig.add_subplot(gridspec[0])
#
#        for i, prototype in enumerate(prototypes_2d):
#            color = colors[self.concepts['labels'][i][0]-1]
#            ax1.plot(prototype[0], prototype[1], '.', color=color)
#            width = self.concepts['thresholds'][i]/pca.singular_values_[0]
#            height = self.concepts['thresholds'][i]/pca.singular_values_[1]
#            ax1.add_artist(Ellipse((prototype[0], prototype[1]),
#                                   width, height,
#                                   color=color, fill=False))
#

@attr.s
class NELTSConceptLearner(NeverEndingConceptLearner):
    """Concept learner NELTS (Towards Never-Ending Learning for Time-Series)

    Parameters
    ----------
    labels: list of str
        the labels for each subsequence. This might not be available to you,
        and should not be an input here. Will be refactored away soon.

    max_subtree_size: int, default=4
        Maximum number of leafs allowed in the subtrees for consideration

    n_top_subtrees: int, default=6
        Number of most significant subtrees to consider

    w: int, default=20
        size of the buffer holding the subsequences for the hierarchical
        clustering

    linkage_method: str, default 'average'
        Method used for the hierarchical clustering. See the `scipy docs
        <https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html>`_
        for a description of the available options.

    n_offline_samples: int, default=1e5
        Number of subsequences sampled for learning the baseline parameters for
        each subtree size

    threshold_factor: float, default=1
        Factor being multiplied with the significant score for setting the
        threshold when defining a new concept.

    query_trigger_threshold: float, default=-norm.ppf(0.7),
        Threshold for when a significance score should trigger a request for a
        label

    Notes
    -------
    .. note:: Differences to the algorithm described in the paper/provided in the MATLAB implementation:

       -  the subsequenceprocessing is only domain-dependent as the
          domain_dependent_processing function is a part of it; an assumption
          was made here that this is already handled beforehand
       - Some parameters have a slightly different meaning here, see the note
         on settings for more details.

    .. note:: Settings compared to the MATLAB script example_activity.m

       *max_subtree_size* (aka overflow_num) is set to 6 in the
       example_activity.m script, but the MATLAB implementation counts all
       nodes in the subtree except the root.
       E.g. a subtree with 4 leafs then have 7-1 nodes.
       Here we simply count the leafs and restrict them

       *n_top_subtrees* corresponds to *K* mentioned briefly in the paper

       The default values here follow the example_activity.m or hardcoded
       values in the MATLAB implementation


    .. attention:: When considering the matlab implementation, be aware of

       # . The implementation uses the linkage method 'single' for offline training
          instead of using 'average' as they do in the online algorithm.
       # . The "permutations" performed are flipping of the given
          subsequence either horizontially or vertically. So the sequences are
          not scrambled as one might expect. This version of the python
          implementation performs the same "permutations".
       # . The paper states that the threshold used for the concepts is
          three times the height of top subtree, but it is computed incorrectly.
          See line 98 in "mypatternfinder_multilabels_v4.mat".
          Also note it only uses a factor of 1 multiplied with the incorrect
          height of the top subtree in the example_activity.mat script.

    """
    cluster_buffer = attr.ib()
    frequent_pattern_maintenance = attr.ib()
    active_learning_system = attr.ib()
    concepts = attr.ib(default={'labels': [], 'thresholds': [], 'counts': [
    ], 'prototypes': []})  # Concepts are clusters with a label
    _nearest_neighbors = attr.ib(
        init=False, default=NearestNeighbors(n_neighbors=1))

    def fit(self, data_point, label):
        top, self.cluster_buffer = self.frequent_pattern_maintenance.evaluate(
            data_point, self.cluster_buffer)
        self.concepts, self.cluster_buffer = self.active_learning_system.evaluate(
            top, self.concepts, self.cluster_buffer, label)

    def is_known(self, data_point):
        known = False

        try:
            prototypes = np.vstack(self.concepts['prototypes'])
            self._nearest_neighbors.fit(prototypes)

            dist, index = self._nearest_neighbors.kneighbors(data_point)
            dist = dist[0][0]
            index = index[0][0]

            if dist < self.concepts['thresholds'][index]:
                # print('An instance of class {} was detected'.format(
                #    concept_dict['labels'][index]))
                self.concepts['counts'][index] += 1
                known = True
        except ValueError:
            # print('No prototypes learned yet')
            pass

        self.was_known = known
        return known

    def plot_clusters(self):
        if self.was_known:
            return False

        HC = self.frequent_pattern_maintenance.hierarchical_clustering
        success = False
        try:
            top = HC.most_significant_subtree(self.cluster_buffer)

            gs0 = gridspec.GridSpec(1, 2)
            gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0])
            gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1])
            gs010 = gridspec.GridSpecFromSubplotSpec(
                10, 1, subplot_spec=gs01[0])
            gs011 = gridspec.GridSpecFromSubplotSpec(
                1, 1, subplot_spec=gs01[1])
            # gs02 = gridspec.GridSpecFromSubplotSpec(20, 5, subplot_spec=gs0[2])

            gs001 = gridspec.GridSpecFromSubplotSpec(
                self.cluster_buffer.shape[0], 1, subplot_spec=gs00[1], wspace=0.35)

            fig = plt.figure(figsize=(15, 15))
            ax_dendogram = fig.add_subplot(gs00[0])
            dn = hierarchy.dendrogram(
                HC.Z, orientation='left', get_leaves=True, ax=ax_dendogram)

            ax_leafs = []
            steps = int(20/(self.cluster_buffer.shape[0]))
            for i in range(self.cluster_buffer.shape[0]):
                ax_leafs.append(fig.add_subplot(
                    gs001[i, :]))

            for i, lidx in enumerate(dn['leaves']):
                ax_leafs[i].plot(self.cluster_buffer[lidx, :])
                ax_leafs[i].axis('off')

            ax_clusters = []
            for i, subtree in enumerate(HC.nonoverlapping_subtrees):
                ax_clusters.append(fig.add_subplot(gs010[i:(i+1), :]))
                for leaf in subtree['leafs']:
                    ax_clusters[i].plot(
                        self.cluster_buffer[leaf, :], 'b')

                mean_pattern = np.mean(
                    self.cluster_buffer[subtree['leafs'], :], axis=0)

                ax_clusters[i].plot(mean_pattern, 'r', linewidth=5)

            try:
                self.plot_concepts(gs011)
            except ValueError:
                pass
            success = True

        except ValueError:
            pass

        return success

    def plot_concepts(self, gridspec):
        pca = PCA(n_components=2)
        prototypes = np.vstack(self.concepts['prototypes'])
        prototypes_2d = pca.fit_transform(prototypes)

        colors = 'rgby'  # TODO: make this dynamic

        #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        fig = plt.gcf()
        ax1 = fig.add_subplot(gridspec[0])

        for i, prototype in enumerate(prototypes_2d):
            color = colors[self.concepts['labels'][i][0]-1]
            ax1.plot(prototype[0], prototype[1], '.', color=color)
            width = self.concepts['thresholds'][i]/pca.singular_values_[0]
            height = self.concepts['thresholds'][i]/pca.singular_values_[1]
            ax1.add_artist(Ellipse((prototype[0], prototype[1]),
                                   width, height,
                                   color=color, fill=False))

        # for i, label in enumerate(np.unique(self.concept_dict['labels'])):
        #    for idx in np.where(self.concept_dict
        #        ax2.plot(prototype)
        #    ax2.plot(prototype)
        #    ax2.title(self.concept_dict['labels'][i])


class FrequentPatternMaintenance:
    """
    Functionality:
        - pre-train offline
        - evaluate new subsequences compared to the buffer

           Question(1): Do they only extract the subtrees of a certain size or
           compute subtrees for all subsets of data?
           Answer(1): They only consider subtrees of size maxSubtreeSize and
           partition in the tree into non-overlapping subtrees sorted by
           their score
           Question(2): Is max_subtree_size the number of all nodes in a subtree
           or is it the number of leaf nodes?
           Answer(2): All nodes in the subtree minus 1 (the root)
    """

    def __init__(self, w=20, max_subtree_size=4, n_top_subtrees=6,
                 linkage_method='average'):
        # Set parameters
        self.w = w  # Size of hierarchical_clustering (memory)
        self.max_subtree_size = max_subtree_size
        self.n_top_subtrees = n_top_subtrees

        self.hierarchical_clustering = HierarchicalClustering(
            self.w, self.max_subtree_size, self.n_top_subtrees, linkage_method)

    def fit_offline(self, sample_sequence, sequence_length, n_samples):
        self.hierarchical_clustering.fit_offline(
            sample_sequence, sequence_length, n_samples)

    def evaluate(self, subsequence, frequent_pattern_buffer):

        frequent_pattern_buffer = self.update_buffer(subsequence,
                                                     frequent_pattern_buffer)

        if frequent_pattern_buffer.shape[0] < self.w:
            top = None
        else:
            # x = np.vstack(frequent_pattern_buffer)
            x = frequent_pattern_buffer
            top = self.hierarchical_clustering.most_significant_subtree(x)

        return top, frequent_pattern_buffer

    def update_buffer(self, subsequence, frequent_pattern_buffer):

        if frequent_pattern_buffer.shape[0] < self.w:
            frequent_pattern_buffer = np.vstack(
                [frequent_pattern_buffer, subsequence])
        else:
            # determine what to replace
            top_leafs = self.hierarchical_clustering.top_leafs
            if len(top_leafs) == self.w:
                # TODO: investigate if the last leaf is the least tight node
                frequent_pattern_buffer[top_leafs[-1], :] = subsequence
            else:
                bottom_leafs = list(set(range(self.w)).difference(top_leafs))
                rm_idx = np.random.choice(bottom_leafs)

                frequent_pattern_buffer[rm_idx, :] = subsequence

        return frequent_pattern_buffer


class ActiveLearningSystem:

    def __init__(self, query_trigger_threshold, threshold_factor):
        self.query_trigger_threshold = query_trigger_threshold
        self.threshold_factor = threshold_factor
        self._request_label_function = None

    def evaluate(self, top, concept_dict, frequent_pattern_buffer, label):

        if top is not None and top['score'] >= self.query_trigger_threshold:
            print('new concept!')
            prototype = np.mean(
                frequent_pattern_buffer[top['leafs'], :], axis=0)

            concept_dict['labels'].append(label)
            concept_dict['counts'].append(0)
            concept_dict['prototypes'].append(prototype)
            concept_dict['thresholds'].append(
                top['dist']*self.threshold_factor)

            frequent_pattern_buffer = np.delete(frequent_pattern_buffer,
                                                top['leafs'], axis=0)
        elif top is not None:
            print('score {} < {}'.format(
                top['score'], self.query_trigger_threshold))

        return concept_dict, frequent_pattern_buffer
