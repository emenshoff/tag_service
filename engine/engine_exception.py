
import logging

log = logging.getLogger('engine')



class EngineExceptionCritical(Exception):
    ''' Citical exception, engine fail '''
    def __init__(self, msg):
        super().__init__(msg)
        log.critical(msg)


class EngineExceptionError(Exception):
    ''' Engine error, not fatal '''
    def __init__(self, msg):
        super().__init__(msg)
        log.error(msg)