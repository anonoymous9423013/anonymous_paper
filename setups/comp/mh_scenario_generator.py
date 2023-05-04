from random import randint
from code_pipeline.tests_generation import RoadTestFactory
from time import  sleep
import numpy as np
from sample_test_generators.mh_killer import killBeamNG
from code_pipeline.validation import TestValidator

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

    def addPoint(self, road_points, validator, n_max=10):
        log.info(f"MH: Adding point to road {road_points}")
        for i in range(n_max):
            log.info(f"MH: Attempt {i} of {n_max}")
            rp = road_points.copy()
            rp.append((randint(0, self.map_size), randint(0, self.map_size)))
            the_test = RoadTestFactory.create_road_test(rp)
            if validator.validate_test(the_test)[0]:
                log.info('MH: Added a valid point...')
                return True, rp
        return False, road_points


    def start(self, n_tests=1000):
        output = 'scenarios.csv'
        mh_validator = TestValidator(self.map_size)
        cnt = 0
        while cnt < n_tests:
            try:
                print(f'Roads generated: {cnt} out of {n_tests}')
                # Pick up random points from the map. They will be interpolated anyway to generate the road
                sleep(0.5)
                road_points = []
                flag_valid = True
                x, y = 10, 10
                road_points.append((x, y))
                for i in range(3):
                    is_valid, rp = self.addPoint(road_points, mh_validator)
                    if is_valid:
                        road_points = rp.copy()
                    else:
                        log.info("MH: Invalid test. Skipping...")
                        flag_valid = False
                        break
                if not flag_valid:
                    continue
                print('*' * 100)
                log.info("Generated test using: %s", road_points)
                # Decorate the_test object with the id attribute
                the_test = RoadTestFactory.create_road_test(road_points)

                ##  MH: Check validity of the test...
                if not mh_validator.validate_test(the_test)[0]:
                    log.info("MH: Invalid test. Skipping...")
                    continue

                # Save the test to the output file
                with open(output, 'a') as f:
                    for point in road_points:
                        f.write(f'{point[0]},{point[1]},')
                    f.write('\n')
                cnt -=- 1
            except Exception as e:
                log.error(e)
                sleep(5)
                continue

if __name__ == "__main__":
    mht = MHTestGenerator(map_size=300)
    mht.start(n_tests=200)