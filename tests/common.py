import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class TestException(Exception):
   def __init__(self, msg):
        super().__init__(msg)
        log.error(msg)