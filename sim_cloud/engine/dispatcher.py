from log.logger import SimLogger

logger = SimLogger.get_logger("DISPATCHER")


class Dispatcher:
    def __init__(self, state):
        self.state = state
        
    def dispatch(self, event):
        """
        Apply one event to simulation state.
        """
        # logger.info(
        #         f"[DISPATCH] t={event.time} | "
        #         f"event={event.__class__.__name__} | "
        #         f"id={id(event)}"
        #     )
        
        try:
            result = event.apply(self.state)
            return result
        except Exception as e:
            logger.error(
                f"[DISPATCH-ERROR] t={event.time} | "
                f"event={event.__class__.__name__} | "
                f"error={e}"
            )
            raise