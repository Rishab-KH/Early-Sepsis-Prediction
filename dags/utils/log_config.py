def setup_logging(PROJECT_ROOT, name):
    
    """
    Set up a logger for the application.

    This function configures a logger that can be used to log messages to the console and a log file. The log file is
    rotated to keep a limited number of log files to prevent them from growing too large.

    :param PROJECT_ROOT: A Path object representing the project's root directory.
    :param name: A string representing the logger's name.
    :return: A configured logger object.
    """

    import logging
    import logging.handlers
    from pathlib import Path
    # Determine the project root and log directory
    # PROJECT_ROOT = Path(__file__).parent.parent.parent.parent # Assuming this is the correct project structure
    log_dir = PROJECT_ROOT + '/logs'
    log_filepath = log_dir + '/Sepsis_Detections_logs.log'  # The path for the log file
    print(PROJECT_ROOT) # Print the determined project root for debugging purposes
    # Ensure the logs directory exists
    # log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.handlers.RotatingFileHandler(
        log_filepath, maxBytes=10*1024*1024, backupCount=3  # 10MB per file, keep last 3 logs
    )
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    return logger