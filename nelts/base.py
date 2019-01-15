from abc import ABC, abstractmethod
import attr
from IPython.display import clear_output, display
from ipywidgets import widgets
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import animation, rcParams
from tqdm import tqdm
import logging
from datetime import datetime
import sys
from pathlib import Path


#rcParams['figure.dpi'] = 150


@attr.s
class DataStream(ABC):

    def stream(self):
        return NotImplemented


@attr.s
class DataStreamPreprocessor(ABC):

    @abstractmethod
    def __call__(self, data_point):
        return NotImplemented


@attr.s
class Concept:
    prototype = attr.ib()
    count = attr.ib(default=0)
    label = attr.ib(default='')


@attr.s
class NeverEndingConceptLearner(ABC):
    """ Interface for a clustering method used by the NeverEndingLearner
    """
    cluster_buffer = attr.ib()
    concepts = attr.ib()  # Concepts are clusters with a label

    @abstractmethod
    def fit(self, data_point):
        return NotImplemented

    @abstractmethod
    def is_known(self, data_point):
        return NotImplemented

    @abstractmethod
    def plot_clusters(self):
        return NotImplemented

    @abstractmethod
    def plot_concepts(self):
        return NotImplemented

    def evaluate(self, data_point, label):
        if not self.is_known(data_point):
            self.fit(data_point, label)


@attr.s
class NeverEndingFramework:
    data = attr.ib()
    preprocessor = attr.ib()

    concept_learner = attr.ib()
    verbose = attr.ib(default=True)
    plots = attr.ib(factory=list)
    output_mode = attr.ib(default='save')
    output_widget = attr.ib(
        default=widgets.Output(layout={'height': '550px'}))

    def run(self):
        max_it = 100
        for data_point, label in tqdm(self.data.stream()):
            #print('processing data {}'.format(self.data.current_idx))
            preprocessed_data_point = self.preprocessor(data_point)
            self.concept_learner.evaluate(preprocessed_data_point, label)
            self.capture_plot()
            # if self.data.current_idx > max_it:
            #    breakpoint()
            #    max_it += 100

    def capture_plot(self):
        # self.output_widget.clear_output(wait=True)
        # with self.output_widget:
        status_plots = self.concept_learner.plot_clusters()
        if status_plots:
            if self.output_mode == 'save':
                plt.savefig(
                    'figures/plot_{}.jpg'.format(self.data.current_idx))
                plt.close()
            elif self.output_mode == 'inline':
                plt.show()
            else:
                logging.debug('No output chosen, closing figure.')
                plt.close()

    def plot_run(self):
        ani = animation.ArtistAnimation(
            self.fig, self.plots, interval=450, blit=True)
        plt.show()


# if __name__ == "__main__":
#
#    LOGPATH = Path('logs')
#
#    LOGPATH.mkdir(parents=True, exist_ok=True)
#
#    NOW = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
#    LOGGING_LEVEL = logging.INFO
#    logging.basicConfig(
#        level=LOGGING_LEVEL,
#        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#        handlers=[
#            logging.FileHandler(LOGPATH/'{}.log'.format(NOW)),
#            logging.StreamHandler(sys.stdout)
#        ])
