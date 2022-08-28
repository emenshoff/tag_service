import sys
import os
import logging


logging.basicConfig(
                    filename='tests.log', 
                    filemode='w',
                    level=logging.DEBUG,

                    format="%(asctime)s: %(module)s: %(levelname)s: %(message)s",
                    datefmt='%d-%b-%y %H:%M:%S'
                    )

log = logging.getLogger(__name__)