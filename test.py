from __future__ import absolute_import

from got10k.experiments import *
from UDT import UDTracker

if __name__ == '__main__':
    tracker = UDTracker()

    root_dir='/home/airlab/PycharmProjects/pythonProject5/data/OTB2013'
    e = ExperimentOTB(root_dir, version=2013)
    e.run(tracker,visualize=False)
    e.report([tracker.name])
