import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_log(av):
    if av:
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

        if not os.path.isdir(os.path.dirname(av.logpath)):
            os.makedirs(os.path.dirname(av.logpath))
        handler = logging.FileHandler(av.logpath, "w")
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(handler)
