import random
import logging as log

log.basicConfig(level=log.INFO)

class MHConfig:
    def __init__(self, target_index=10, time_of_day=0.5, ego_speed=30, traffic_amount=10, keep_lane=0, weather=0):
        self.target_index = target_index
        self.time_of_day = time_of_day
        self.ego_speed = ego_speed
        self.traffic_amount = traffic_amount
        self.keep_lane = keep_lane
        self.weather = weather
    
    def __str__(self):
        return f'{self.target_index}, {self.time_of_day}, {self.ego_speed}, {self.traffic_amount}, {self.keep_lane}, {self.weather}'

    def fromStr(self, config_str):
        config = config_str.strip().rstrip().split(',')
        config = [x.strip().rstrip() for x in config]
        self.target_index = int(config[0])
        self.time_of_day = float(config[1])
        self.ego_speed = float(config[2])
        self.traffic_amount = int(config[3])
        self.keep_lane = int(config[4])
        self.weather = int(config[5])
        return self

    def generateRandomConfig(self):
        self.target_index = random.randint(0, 50)
        self.time_of_day = round(random.random(), 2)
        self.ego_speed = random.randint(20, 50)
        self.traffic_amount = random.randint(0, 10)
        self.keep_lane = random.randint(0, 1)
        self.weather = random.randint(0, 6)

if __name__ == '__main__':
    config = MHConfig()
    output_file = 'config.txt'
    n_configs = 1000
    configs = []
    for i in range(n_configs):
        config.generateRandomConfig()
        configs.append(str(config))
    configs.sort()
    with open(output_file, 'w') as f:
        for config in configs:
            f.write(config + '\n')
        f.write('done')
            
    log.info(f'Generated {n_configs} configs and saved to {output_file}!')