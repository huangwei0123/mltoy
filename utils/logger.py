import logging
import sys


class LoggerFactory:

    @staticmethod
    def create(log_file):

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        return logging.getLogger(__name__)

