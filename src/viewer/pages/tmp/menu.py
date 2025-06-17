import utils.utils_pages as utilpg

from utils.utils_logger import setup_logger
logger = setup_logger()

logger.debug('Start of Menu Screen!')

# Page config should be called for each page
utilpg.config_page()
utilpg.show_menu()
