import logging

# create logger with 'spam_application'
logger = logging.getLogger('chi')
logger.propagate = False
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
# fh = logging.FileHandler('spam.log')
# fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('|%(asctime)s| chi/%(module)s: %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
# logger.addHandler(fh)
logger.addHandler(ch)


def set_loglevel(level: str):
  l = getattr(logging, level.upper())
  logger.setLevel(l)

if __name__ == "__main__":
  logger.info('info')
  logger.debug('debug')


