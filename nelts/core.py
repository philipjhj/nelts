from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import zscore, norm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from . hierarchical_clustering import HierarchicalClustering
from . extra import get_label_from_data

from pdb import set_trace


class SubsequenceProcessing:

    def __init__(self):
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1)

    def detect(self, subsequence, concept_dict):

        try:
            prototypes = np.vstack(concept_dict['prototypes'])
        except ValueError:
            # print('No prototypes learned yet')
            return False, concept_dict

        self.nearest_neighbors.fit(prototypes)

        dist, index = self.nearest_neighbors.kneighbors(subsequence)
        dist = dist[0][0]
        index = index[0][0]

        if dist < concept_dict['thresholds'][index]:
            # print('An instance of class {} was detected'.format(
            #    concept_dict['labels'][index]))
            concept_dict['counts'][index] += 1
            return True, concept_dict
        return False, concept_dict


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

    def __init__(self, w=20, max_subtree_size=4, n_top_subtrees=6, linkage_method='average',
                 output_mode='inline'):
        # Set parameters
        self.w = w  # Size of hierarchical_clustering (memory)
        self.max_subtree_size = max_subtree_size
        self.n_top_subtrees = n_top_subtrees
        self.output_mode = output_mode
        self.eval_counter = 0

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

            if (hasattr(self.hierarchical_clustering, 'Z') and
                    self.output_mode in ['inline', 'save']):
                self.hierarchical_clustering.plot_status(x)
                if self.output_mode == 'save':
                    plt.savefig(
                        'figures/plot_{}.jpg'.format(self.eval_counter))
                else:
                    plt.show()

                self.eval_counter += 1

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

    @property
    def request_label_function(self):
        return self._request_label_function

    @request_label_function.setter
    def request_label_function(self, request_label_function):
        self._request_label_function = request_label_function

    def evaluate(self, top, concept_dict, frequent_pattern_buffer, position):

        if top is not None and top['score'] >= self.query_trigger_threshold:
            # print('score {}'.format(top['score']))
            # print('new concept!')
            prototype = np.mean(
                frequent_pattern_buffer[top['leafs'], :], axis=0)
            label = self.request_label(position, prototype)

            concept_dict['labels'].append(label)
            concept_dict['counts'].append(0)
            concept_dict['prototypes'].append(prototype)
            concept_dict['thresholds'].append(
                top['dist']*self.threshold_factor)

            frequent_pattern_buffer = np.delete(frequent_pattern_buffer,
                                                top['leafs'], axis=0)

        return concept_dict, frequent_pattern_buffer

    def request_label(self, position, prototype):
        #print('requesting label')
        return self._request_label_function(position, prototype)


class SequenceData:

    def __init__(self, sequence, subsequence_length, diff_threshold=None):
        if np.ndim(sequence) == 2:
            sequence = sequence.reshape(1, -1)
        self.sequence = sequence
        self.subsequence_length = subsequence_length
        self.diff_threshold = diff_threshold


class NELTS:
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

    output_mode: str, default = 'inline' ('save' for saving locally)
        Setting for choosing what kind of output should be given.

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

       #. The implementation uses the linkage method 'single' for offline training
          instead of using 'average' as they do in the online algorithm.
       #. The "permutations" performed are flipping of the given
          subsequence either horizontially or vertically. So the sequences are
          not scrambled as one might expect. This version of the python
          implementation performs the same "permutations".
       #. The paper states that the threshold used for the concepts is
          three times the height of top subtree, but it is computed incorrectly.
          See line 98 in "mypatternfinder_multilabels_v4.mat".
          Also note it only uses a factor of 1 multiplied with the incorrect
          height of the top subtree in the example_activity.mat script.


    """

    def __init__(self, max_subtree_size=4, n_top_subtrees=6, w=20,
                 linkage_method='average', n_offline_samples=1e5,
                 threshold_factor=1, query_trigger_threshold=-norm.ppf(0.7),
                 output_mode='inline'):

        self.settings = {key: value for key, value in locals().items() if key
                         not in ['self', 'labels']}

        self.subsequence_processing = SubsequenceProcessing()

        self.frequent_pattern_maintenance = FrequentPatternMaintenance(
            w=w, max_subtree_size=max_subtree_size,
            n_top_subtrees=n_top_subtrees,
            linkage_method=linkage_method,
            output_mode=output_mode)

        self.active_learning_system = ActiveLearningSystem(
            query_trigger_threshold=query_trigger_threshold,
            threshold_factor=threshold_factor)

        # list of dicts with concept information
        self.concept_dict = {'labels': [], 'thresholds': [], 'counts': [],
                             'prototypes': []}

    def never_ending_learning(self, subsequence_length, S=None, labeling_function=None):
        """ The main loop evaluating the data stream

        Parameters
        ----------
        subsequence_length : int
            Length of the subsequences to extract from the data stream
        S : np.array with shape (1, N)
            The data stream array of length N.
        labeling_function: callable, or array of labels
            The function used to provide label information for data points as
            determined by the active learning system. If labels are known for
            the whole stream, they can be provided as an array as well.
        """

        if S is not None:
            self.S = SequenceData(S, subsequence_length)
        else:
            print('No data given...')
            return

        if callable(labeling_function):
            self.active_learning_system.request_label_function = labeling_function
        elif len(labeling_function) == self.S.sequence.shape[1]:
            # When the length of the stream and labeling information is
            # available (for testing/experiments)
            self.active_learning_system.request_label_function = lambda x, y: get_label_from_data(
                labeling_function, x, y)
        else:
            print('No labeling function is provided...')
            return

        self.initialize_learning()
        self.frequent_pattern_buffer = np.empty((0, subsequence_length))

        for sub, pos in self.get_next_subsequence():
            # print(pos)
            try:
                detected, self.concept_dict = self.subsequence_processing.detect(
                    sub, self.concept_dict)
                if not detected:
                    top, self.frequent_pattern_buffer = self.frequent_pattern_maintenance.evaluate(
                        sub, self.frequent_pattern_buffer)
                    self.concept_dict, self.frequent_pattern_buffer = self.active_learning_system.evaluate(
                        top, self.concept_dict, self.frequent_pattern_buffer,
                        pos)
            except KeyboardInterrupt:

                break

        print('Stream is empty. Plotting learned concepts')

        self.plot_concept_dict()

    def initialize_learning(self):
        """ pre-trains the frequentpatternmaintenance algorithm and finds a
        suitable threshold for determining when a new subsequence has arrived
        """

        self.frequent_pattern_maintenance.fit_offline(
            self.S.sequence, self.S.subsequence_length,
            self.settings['n_offline_samples'])

        # Current version assumes this is good enough; improvement could be
        # manually requesting a threshold as in the MATLAB version
        self.S.diff_threshold = self.frequent_pattern_maintenance.hierarchical_clustering.baseline_mean[
            0]

    def get_next_subsequence(self):

        current_subsequence = np.repeat(
            [float('inf')], self.S.subsequence_length)

        for i in range(self.S.sequence.shape[1]-self.S.subsequence_length):
            next_candidate_subsequence = zscore(
                self.S.sequence[0, i:i+self.S.subsequence_length], ddof=1)

            euc_dist = np.linalg.norm(
                current_subsequence-next_candidate_subsequence)

            if euc_dist > self.S.diff_threshold:
                current_subsequence = next_candidate_subsequence.reshape(1, -1)
                yield current_subsequence, i

    def plot_concept_dict(self):

        pca = PCA(n_components=2)
        prototypes = np.vstack(self.concept_dict['prototypes'])
        prototypes_2d = pca.fit_transform(prototypes)

        colors = 'rgby'  # TODO: make this dynamic

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

        for i, prototype in enumerate(prototypes_2d):
            color = colors[self.concept_dict['labels'][i]-1]
            ax1.plot(prototype[0], prototype[1], '.', color=color)
            width = self.concept_dict['thresholds'][i]/pca.singular_values_[0]
            height = self.concept_dict['thresholds'][i]/pca.singular_values_[1]
            ax1.add_artist(Ellipse((prototype[0], prototype[1]),
                                   width, height,
                                   color=color, fill=False))

        # for i, label in enumerate(np.unique(self.concept_dict['labels'])):
        #    for idx in np.where(self.concept_dict
        #        ax2.plot(prototype)
        #    ax2.plot(prototype)
        #    ax2.title(self.concept_dict['labels'][i])

        plt.show()
