import logging
import os
import sys
import os.path as osp
def setup_logger(name, save_dir, if_train, local_rank=0):
    logger = logging.getLogger(name)

    level = logging.INFO if local_rank in [-1, 0] else logging.WARN
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(level=level)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)


    formatter = logging.Formatter("%(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger