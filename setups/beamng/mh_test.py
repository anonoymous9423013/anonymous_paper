from beamngpy import BeamNGpy, Scenario, Road, Vehicle
from beamngpy.sensors import Ultrasonic, State, Damage, Lidar
import numpy as np
from mh_config import MHConfig
from mh_tester import MHTester
import logging as log
import pprint
import os

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BNG_HOME = "C:\\Research\\BeamNG.tech.v0.27.2.0"



def executeTest(bng, config, n=0):
    index_target_waypoint = config.target_index
    time_of_day = config.time_of_day
    speed = config.ego_speed
    traffic_amount = config.traffic_amount
    keep_lane = config.keep_lane
    weather = ['cloudy_evening', 'sunny_noon', 'sunny_evening',
                'foggy_morning', 'foggy_night', 'sunny', 'rainy'][config.weather]
    ##  Create a scenario...
    scenario = Scenario('west_coast_usa', 'SEDNA')
    
    ##  Create an ETK800 with the licence plate 'PYTHON'
    ego = Vehicle('ego', model='etk800', color='White', license='SEDNA')
    ##  Add it to our scenario at this position and rotation
    scenario.add_vehicle(ego, pos=(-725.711, 554.32, 120.26),
                        rot_quat=(-0.0064, 0.0008, -5301, 0.8478))
    ##  Place files defining our scenario for the simulator to read
    scenario.make(bng)
    ##  Load and start our scenario
    bng.scenario.load(scenario)
    bng.settings.set_deterministic(60)  #  60 Hz
    ##  Sensors...
    ultrasonic = Ultrasonic('ultrasonic', bng, ego)
    damage = Damage()
    state = State()
    lidar = Lidar("lidar", bng, ego, is_visualised=False)
    ego.sensors.attach('damage', damage)
    sensors = [ultrasonic, damage, state, lidar]
    ##  Choose source and target waypoints...
    waypoints = {w.name: w for w in scenario.find_waypoints()}
    target = list(waypoints.keys())[index_target_waypoint]
    target_pos = waypoints[target].pos
    # ego.teleport(source.pos)
    ego.switch()
    bng.scenario.start()
    bng.traffic.spawn(max_amount=traffic_amount)
    bng.env.set_tod(time_of_day)
    bng.env.set_weather_preset(weather)
    ego.ai.set_mode('manual')
    ego.ai.set_waypoint(target)
    ego.ai.drive_in_lane(keep_lane)
    ego.ai.set_speed(speed)
    for i in range(72):
        print('Step: ', i)
        bng.step(30)
        sensor_data = readSensors(ego, sensors)
        logData(sensor_data, target_pos, config, n)

def logData(sensor_data, target_pos, config, n):
    f_ultrasound = sensor_data[0]
    f_damage = sensor_data[1]
    f_dist = np.linalg.norm(sensor_data[2] - target_pos)
    f_lidar = np.min(sensor_data[3][sensor_data[3] > 0])
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, str(config)+f'_{n}.txt')
    with open(output_file, 'a') as f:
        f.write('*'*50 + '\n')
        f.write('Ultrasound: ' + str(f_ultrasound) + '\n')
        f.write('Damage: ' + str(f_damage) + '\n')
        f.write('Distance: ' + str(f_dist) + '\n')
        f.write('Lidar: ' + str(f_lidar) + '\n')


def readSensors(ego, sensors):
    ego.sensors.poll()
    ultrasonic = sensors[0]
    lidar = sensors[-1]
    ultrasonic_data = ultrasonic.poll()['distance']
    lidar_data = lidar.poll()['pointCloud']
    damage_data = ego.sensors['damage']['damage']
    state_data = ego.sensors['state']
    current_pos = np.array(state_data['pos'])
    sensor_data = [ultrasonic_data, damage_data, current_pos, lidar_data]
    return sensor_data

if __name__ == '__main__':
    n_repeats = 10
    input_file = 'config.txt'
    output_file = 'output.txt'
    tester = MHTester(n_repeats=n_repeats)
    config_str = tester.findNextConfig()
    print('Next config:', config_str)
    print('No of repeats:', tester.findNoOfRepeats(config_str))
    with BeamNGpy('localhost', 64256, home=BNG_HOME) as beamng:
        while tester.findNextConfig() != 'done':
            config_str = tester.findNextConfig()
            log.info(f'Starting new test for config {config_str}...')
            config = MHConfig().fromStr(config_str)
            bng = beamng.open(launch=True)
            n = tester.findNoOfRepeats(config_str)
            executeTest(bng, config, n=n)
            with open(output_file, 'a') as f:
                f.write(f'{config_str}_{n}\n')
            log.info(f'Finished repetition {n} for {config_str}...')
