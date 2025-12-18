from abc import ABC, abstractmethod
class BaseScheduler(ABC):
    @abstractmethod
    def on_time_step(self, state):
        """return list of events to be scheduler
        """
        pass