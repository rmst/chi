import logging

# create logger with 'spam_application'
logger = logging.getLogger('rl-flow')
logger.propagate = False
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
# fh = logging.FileHandler('spam.log')
# fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('|%(asctime)s| rl-flow/%(module)s: %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
# logger.addHandler(fh)
logger.addHandler(ch)
if __name__ == "__main__":
  logger.info('info')
  logger.warn('warn')
  logger.debug('debug')
