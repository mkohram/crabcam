from threading import Thread
import queue
import time
import logging
import os

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class TaskQueue(queue.Queue):

    def __init__(self, num_workers=1):
        queue.Queue.__init__(self)
        self.num_workers = num_workers
        self.start_workers()

    def add_task(self, task, *args, **kwargs):
        args = args or ()
        kwargs = kwargs or {}
        self.put((task, args, kwargs))

    def start_workers(self):
        for i in range(self.num_workers):
            t = Thread(target=self.worker)
            t.daemon = True
            t.start()

    def worker(self):
        try:
            while True:
                item, args, kwargs = self.get()
                item(*args, **kwargs)
                logger.info('task finished')
                self.task_done()
        except Exception as e:
            # TODO need to do some reporting here
            pass

    def join_with_timeout(self, timeout):
        self.all_tasks_done.acquire()
        try:
            endtime = time.time() + timeout
            while self.unfinished_tasks:
                remaining = endtime - time.time()
                if remaining <= 0.0:
                    raise NotFinished
                self.all_tasks_done.wait(remaining)
        finally:
            self.all_tasks_done.release()

def tests():
    def blokkah(*args, **kwargs):
        time.sleep(5)
        print("Blokkah mofo!")

    q = TaskQueue(num_workers=5)

    for item in range(10):
        q.add_task(blokkah)

    q.join()       # block until all tasks are done
    print("All done!")


if __name__ == "__main__":
    tests()
