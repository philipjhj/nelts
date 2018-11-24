from abc import ABC, abstractmethod


class Subsequence:
    def __init__(self, data, original_index):
        self.data = data
        self.original_index = original_index


class ActiveLearner(ABC):

    def set_knowledge_base(self, knowledge_base):
        self.knowledge_base = knowledge_base

    @abstractmethod
    def evaluate(self, subsequence):
        return NotImplemented


class KnowledgeBase(ABC):

    @abstractmethod
    def evaluate(self, subsequence):
        return NotImplemented
