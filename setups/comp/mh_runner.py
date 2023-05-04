from random import randint
from code_pipeline.tests_generation import RoadTestFactory
from time import  sleep
import shutil as sh
from sample_test_generators.mh_counter import nextScenario
import os

import logging as log


class MHTestGenerator():
    """
        This simple (naive) test generator creates roads using 4 points randomly placed on the map.
        We expect that this generator quickly creates plenty of tests, but many of them will be invalid as roads
        will likely self-intersect.
    """

    def __init__(self, executor=None, map_size=None):
        self.executor = executor
        self.map_size = map_size

    def emptyCache(self):
        levels_folder = r'C:\Research\User\0.26'
        sh.rmtree(levels_folder, ignore_errors=True)
        os.makedirs(levels_folder, exist_ok=True)


    def start(self):
        output_folder = 'mh_output'
        scenario_file = 'scenarios.csv'

        while not self.executor.is_over():
            self.emptyCache()
            # Some debugging
            time_remaining = self.executor.get_remaining_time()["time-budget"]
            log.info(f"Starting test generation. Remaining time {time_remaining}")

            # Simulate the time to generate a new test
            sleep(0.5)
            # Pick up random points from the map. They will be interpolated anyway to generate the road
            road_points, cnt = nextScenario(scenario_file, output_folder, n=10)
            road_points = [(road_points[i], road_points[i+1]) for i in range(0, len(road_points), 2)]

            # Some more debugging
            log.info("Generated test using: %s", road_points)
            # Decorate the_test object with the id attribute
            the_test = RoadTestFactory.create_road_test(road_points)

            time_remaining = self.executor.get_remaining_time()["time-budget"]
            log.info(f"Simulated test generation for 0.5 sec. Remaining time {time_remaining}")
            # Try to execute the test
            test_outcome, description, execution_data = self.executor.execute_test(the_test)
            time_remaining = self.executor.get_remaining_time()["time-budget"]
            log.info(f"Executed test {the_test.id}. Remaining time {time_remaining}")

            # Print the result from the test and continue
            log.info("test_outcome %s", test_outcome)
            log.info("description %s", description)


