from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore, norm
import numpy as np
from pdb import set_trace

from . hierarchical_clustering import HierarchicalClustering


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

         Steps:
         0: offline training of the reference subtrees using "pattern-free
         data" of sizes 2 to max_subtree_size
         1:
           Question(1): Do they only extract the subtrees of a certain size or
           compute subtrees for all subsets of data?
           Answer(1): They only consider subtrees of size maxSubtreeSize and
           partition in the tree into non-overlapping subtrees sorted by
           their score
           Question(2): Is max_subtree_size the number of all nodes in a subtree
           or is it the number of leaf nodes?
           Answer(2): ????
    """

    def __init__(self, w=20, max_subtree_size=4, n_top_subtrees=6, linkage_method='average'):
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

            if hasattr(self.hierarchical_clustering, 'Z'):
                self.hierarchical_clustering.plot_current(x)

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


def get_label_from_stream(labels, position):
    return labels[position]


class ActiveLearningSystem:

    def __init__(self, query_trigger_threshold, threshold_factor,
                 request_label=get_label_from_stream, labels=None):
        self.query_trigger_threshold = query_trigger_threshold
        self.threshold_factor = threshold_factor
        self._request_label = get_label_from_stream
        self.labels = labels

    def evaluate(self, top, concept_dict, frequent_pattern_buffer, position):

        if top is not None and top['score'] >= self.query_trigger_threshold:
            # print('score {}'.format(top['score']))
            # print('new concept!')
            label = self.request_label(position)
            prototype = np.mean(
                frequent_pattern_buffer[top['leafs'], :], axis=0)

            concept_dict['labels'].append(label)
            concept_dict['counts'].append(0)
            concept_dict['prototypes'].append(prototype)
            concept_dict['thresholds'].append(
                top['dist']*self.threshold_factor)

            frequent_pattern_buffer = np.delete(frequent_pattern_buffer,
                                                top['leafs'], axis=0)

        return concept_dict, frequent_pattern_buffer

    def request_label(self, position):
        return self._request_label(self.labels, position)


class SequenceData:

    def __init__(self, sequence, subsequence_length, diff_threshold=None):
        if np.ndim(sequence) == 2:
            sequence = sequence.reshape(1, -1)
        self.sequence = sequence
        self.subsequence_length = subsequence_length
        self.diff_threshold = diff_threshold


class NELTS:
    """ python implementation of nelts
        differences to the algorithm described in the paper/provided in the
        matlab implementation:

            - the subsequenceprocessing is only domain-dependent as the
            domain_dependent_processing function is a part of it; an assumption
            that this is already handled beforehand was made here
            -. ...

        concerns about the matlab implementation:
            (1) they use the linkage method 'single' for offline training
            instead of using 'average' as in the online algorithm.
            (2) the "permutations" they perform are flipping of the given
            subsequence either horizontially or vertically. so they are not
            scrambling the sequence as one might expect.
            this version of the python implementation performs the same
            "permutations".
            (3) the paper states that the threshold used for the concepts is
            three times the height of top subtree, but it is computed
            incorrectly. see line 98 in "mypatternfinder_multilabels_v4.mat".
            also note, it uses only one time the incorrect height of the top
            subtree in the example_activity.mat script.

    """

    def __init__(self, labels, domain_dependent_processing, settings={},
                 subsequence_processing=SubsequenceProcessing,
                 frequent_pattern_maintenance=FrequentPatternMaintenance,
                 active_learning_system=ActiveLearningSystem):
        """
        parameters:
            labels: labels for the data stream (not neccessarily available,
            needs refactoring)
            domain_dependent_processing: a function
            settings: dict with the following parameters;
                max_subtree_size:
                w: size of hierarchical_clustering
                n_samples: number of samples of each subtree size to use determine
                    "pattern-free" parameters
                linkage_method: method used in the hierarchical clustering
        """

        self.settings = self.parse_settings(settings)

        self.domain_dependent_processing = domain_dependent_processing

        self.subsequence_processing = subsequence_processing()

        self.frequent_pattern_maintenance = frequent_pattern_maintenance(
            self.settings['w'], self.settings['max_subtree_size'],
            self.settings['n_top_subtrees'],
            self.settings['linkage_method'])

        self.active_learning_system = active_learning_system(
            self.settings['query_trigger_threshold'],
            self.settings['threshold_factor'],
            labels=labels)

        # list of dicts with concept information
        self.concept_dict = {'labels': [], 'thresholds': [], 'counts': [],
                             'prototypes': []}

    def never_ending_learning(self, subsequence_length, S=None, p=None,
                              options=None):

        if S is None and p is not None:
            self.S = self.transform_p_to_s(p)
        elif S is not None:
            self.S = SequenceData(S, subsequence_length)
            self.p = p
        else:
            print('No data given...')
            return

        self.initialize_learning()
        self.frequent_pattern_buffer = np.empty((0, subsequence_length))

        for sub, pos in self.get_next_subsequence_from_S():
            # print(pos)
            detected, self.concept_dict = self.subsequence_processing.detect(
                sub, self.concept_dict)
            if not detected:
                top, self.frequent_pattern_buffer = self.frequent_pattern_maintenance.evaluate(
                    sub, self.frequent_pattern_buffer)
                self.concept_dict, self.frequent_pattern_buffer = self.active_learning_system.evaluate(
                    top, self.concept_dict, self.frequent_pattern_buffer,
                    pos)

        print('Stream is empty')

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

    def get_next_subsequence_from_S(self):

        subsequence_length = self.S.subsequence_length
        full_sequence = self.domain_dependent_processing(self.S.sequence)

        current_subsequence = np.repeat([float('inf')], subsequence_length)

        full_sequence_length = full_sequence.shape[1]
        for i in range(full_sequence_length-subsequence_length):
            next_candidate_subsequence = zscore(
                full_sequence[0, i:i+subsequence_length], ddof=1)

            euc_dist = np.linalg.norm(
                current_subsequence-next_candidate_subsequence)

            if euc_dist > self.S.diff_threshold:
                current_subsequence = next_candidate_subsequence.reshape(1, -1)
                yield current_subsequence, i

    def transform_P_to_S(self, P):
        # Unsure whether this should be here or outside of class
        print('Getting S from P... or not')
        return NotImplemented

    @staticmethod
    def parse_settings(settings):
        # Notes compared to the MATLAB script example_activity.m
        #
        # max_subtree_size (aka overflow_num) is set to 6 in the
        # example_activity.m script, but the MATLAB implementation counts all
        # nodes in the subtree except the root.
        # E.g. a subtree with 4 leafs then have 7-1 nodes.
        # Here we simply count the leafs and restrict them
        #
        # n_top_subtrees corresponds to K mentioned briefly in the paper
        #
        # The default values here follow the example_activity.m or hardcoded
        # values in the MATLAB implementation

        default_settings = {'max_subtree_size': 4,
                            'n_top_subtrees': 6,
                            'w': 20,
                            'linkage_method': 'average',
                            'n_offline_samples': 1e5,
                            'threshold_factor': 1,
                            'query_trigger_threshold': -norm.ppf(0.7)
                            }

        for key, value in default_settings.items():
            if key not in settings.keys():
                print('Using default value for {} (= {}).'.format(key, value))
                settings[key] = value

        return settings
