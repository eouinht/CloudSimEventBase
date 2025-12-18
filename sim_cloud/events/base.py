class Event:
    def __init__(self, time):
        self.time = time

    def apply(self, state, dispatcher):
        raise NotImplementedError
