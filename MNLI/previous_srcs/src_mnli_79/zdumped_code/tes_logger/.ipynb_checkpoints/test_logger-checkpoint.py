import logging
import logging.config

print(__name__)
print(__name__)
# create logger
logging.config.fileConfig('test_logging_config.conf')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

""" if use python code style
# create console handler and set level to debug
ch = logging.StreamHandler()
fh = logging.FileHandler("test_loging.txt")
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(fh)
logger.addHandler(ch)
"""

def hi():
    # 'application' code
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')