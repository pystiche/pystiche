from pystiche.optim import default_logger

logger = default_logger("blub")

logger.debug("debug")
logger.info("info")
logger.warning("warning")
logger.error("error")
logger.error("critical")
