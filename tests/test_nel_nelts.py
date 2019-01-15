from scipy.io import loadmat
from nelts.core import FrequentPatternMaintenance, \
    ActiveLearningSystem, NELTSConceptLearner, NeverEndingFramework, \
    IdentityPreprocessor, SequenceStream
from scipy.stats import norm
import numpy as np

# TODO: Generate such data instead of loading?
originLabelFileName = 'tests/data/stream_labels.mat'
originDataFileName = 'tests/data/stream_data.mat'

labels = loadmat(originLabelFileName)
data = loadmat(originDataFileName)
stream_labels = labels['stream_labels']
stream_data = data['stream_data']

# Load data before here
sequence = stream_data
subsequence_length = 80
print('Data loaded')


def test_nelts():
    # Settings
    max_subtree_size = 4
    n_top_subtrees = 6
    w = 20
    linkage_method = 'average'
    n_offline_samples = 1e2
    output_mode = 'none'
    # Active learning
    threshold_factor = 1
    query_trigger_threshold = -norm.ppf(0.7)

    print('Setting the frequent pattern maintenance')
    frequent_pattern_maintenance = FrequentPatternMaintenance(
        w=w, max_subtree_size=max_subtree_size,
        n_top_subtrees=n_top_subtrees,
        linkage_method=linkage_method)

    print('Fitting frequent pattern maintenance offline')
    frequent_pattern_maintenance.fit_offline(sequence, subsequence_length,
                                             n_offline_samples)

    dist_between_subsequences_threshold = \
        frequent_pattern_maintenance.hierarchical_clustering.baseline_mean[0]

    print('Setting the active learning system')
    active_learning_system = ActiveLearningSystem(
        query_trigger_threshold=query_trigger_threshold,
        threshold_factor=threshold_factor)

    print('Setting the data stream')
    data = SequenceStream(sequence, subsequence_length,
                          dist_between_subsequences_threshold, stream_labels)

    print('Setting the NELTS learner')
    cluster_buffer = np.empty((0, subsequence_length))
    NELTS_learner = NELTSConceptLearner(cluster_buffer=cluster_buffer,
                                        frequent_pattern_maintenance=frequent_pattern_maintenance,
                                        active_learning_system=active_learning_system)

    print('Setting the NELTS framework')
    NELTS = NeverEndingFramework(data=data, preprocessor=IdentityPreprocessor(),
                                 concept_learner=NELTS_learner, output_mode=output_mode)

    NELTS.run()


if __name__ == "__main__":
    test_nelts()
