import logging
import time


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MetricLogger(metaclass=Singleton):
    def __init__(self, level=logging.INFO):
        super().__init__()
        self.level = level
        self.__set_logger(__name__)
        self._experiment = None
        self._experiments = {}
        self.timings = {}
        self.metrics = {}
        self.logger.info('MetricLogger has started')

    def __set_logger(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(c_handler)

    def log_max_time(self, key, start_time):
        elapsed = time.time() - start_time
        self.log_max(key, elapsed)

    def log_max(self, key, value):
        if key in self.timings:
            self.timings[key] = value if value > self.timings[key] else self.timings[key]
        else:
            self.timings[key] = value

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def inc_metric(self, key, val=0):
        if key in self.metrics:
            self.metrics[key] += 1 if val == 0 else val
        else:
            self.metrics[key] = 1 if val == 0 else val

    def append_metric(self, key, val):
        if key in self.metrics:
            if type(self.metrics[key]) == list:
                self.metrics[key].append(val)
            else:
                raise Exception("this is not a list")
        else:
            self.metrics[key] = [val]

    def set_metric(self, key, val):
        self.metrics[key] = val

    def dec_metric(self, key):
        if key in self.metrics:
            self.metrics[key] += 1
        else:
            self.metrics[key] = 1

    def zero_metric(self, key):
        self.metrics[key] = 0

    def flush_metric(self, key):
        self.logger.info("metric {0} : {1}".format(key, self.metrics[key]))

    def flush_all(self):
        for k in self.metrics.keys():
            self.logger.info("metric {0} : {1}".format(k, self.metrics[k]))
        self.flush_timings()

    def flush_timings(self):
        for k in self.timings.keys():
            self.logger.info("timing for {0} : {1}".format(k, self.timings[k]))
