from abc import ABC, abstractmethod

class TraceAdapter(ABC):

    @abstractmethod
    def load_events(self):
        """
        Return: List[Event]
        """
        pass
