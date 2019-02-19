from threading import Thread
import time
from my_functions import get_device_util


class Monitor(Thread):
    def __init__(self, delay, gpu_device_ID):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.gpu_device_ID = gpu_device_ID
        self.gpu_util = -100
        self.start()

    def run(self):
        while not self.stopped:
            self.gpu_util = get_device_util(self.gpu_device_ID)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True
