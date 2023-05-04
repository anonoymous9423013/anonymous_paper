import random
from time import sleep
from mh_config import MHConfig
import os

class MHFlakyTester:
    def __init__(self, test_durations=120, n_scenarios=10, n_runs=10):
        self.test_durations = test_durations
        self.n_scenarios = n_scenarios
        self.n_runs = n_runs
        self.mhc = MHConfig()
        if not os.path.exists('mh_flaky_test_1_done.txt'):
            with open('mh_flaky_test_1_done.txt', 'w') as f:
                f.write('')
        

    def generateNewTest(self):
        self.weather_params = [random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]
        self.weather_params = [int(i) for i in self.weather_params]
        self.ego_vehicle_speed = random.uniform(0, 100)
        self.ego_vehicle_bp_index = random.choice(range(len(self.mhc.getVehicleBlueprints())))
        self.vehicle_bp_index = random.choice(range(len(self.mhc.getVehicleBlueprints())))
        self.n_fronts = random.randint(0, 4)
        self.n_backs = random.randint(0, 3)
        self.n_opposites = random.randint(0, 3)
        return (*self.weather_params, self.ego_vehicle_bp_index, self.vehicle_bp_index, self.n_fronts, self.n_backs, self.n_opposites)

    def saveTestConfigs(self):
        with open('mh_flaky_test_1.txt', 'w') as f:
            for i in range(self.n_scenarios):
                f.write(','.join([str(j) for j in self.generateNewTest()]) + '\n')

    def findLatestTest(self):
        with open('mh_flaky_test_1.txt', 'r') as f:
            scenarios = f.readlines()
        
        with open('mh_flaky_test_1_done.txt', 'r') as f:
            done_scenarios = f.readlines()

        for i in range(len(scenarios)):
            test_index = 0
            scenario = scenarios[i].strip()
            for line in done_scenarios:
                if scenario in line:
                    test_index += 1
            if test_index < self.n_runs:
                return i, test_index
        
        ##  When all tests are done...
        return -1, -1

    def runSingleTest(self, scenario_index, test_index):
        with open('mh_flaky_test_1.txt', 'r') as f:
            lines = f.readlines()
        
        scenario = lines[scenario_index].strip().split(',')
        self.weather_params = [int(i) for i in scenario[:3]]
        self.ego_vehicle_bp_index = int(scenario[3])
        self.vehicle_bp_index = int(scenario[4])
        self.n_fronts = int(scenario[5])
        self.n_backs = int(scenario[6])
        self.n_opposites = int(scenario[7])
        self.ego_vehicle_bp = self.mhc.getVehicleBlueprints()[self.ego_vehicle_bp_index]
        self.vehicle_bp = self.mhc.getVehicleBlueprints()[self.vehicle_bp_index]
    
        del self.mhc
        self.mhc = MHConfig(ego_vehicle_bp=self.ego_vehicle_bp, vehicle_bp=self.vehicle_bp, filename=','.join(scenario)+f'_{test_index}')
        self.mhc.setWeather(*self.weather_params)
        self.mhc.addRelativeVehicle(None, mode='front', n=self.n_fronts, random_bp=False)
        self.mhc.addRelativeVehicle(None, mode='back', n=self.n_backs, random_bp=False)
        self.mhc.addRelativeVehicle(None, mode='opposite', n=self.n_opposites, random_bp=False)
        self.mhc.startTrafficManager()
        self.mhc.run(0.05, self.test_durations)
        print('Test finished!')
        
        with open('mh_flaky_test_1_done.txt', 'a') as f:
            f.write(','.join([str(i) for i in scenario]) + f'_{test_index}' + '\n')

    def run(self):
        while True:
            try:
                scenario_index, test_index = self.findLatestTest()
                print(f'Scenario index: {scenario_index}, Test index: {test_index}')
                if scenario_index == -1:
                    break
                self.runSingleTest(scenario_index, test_index)
            except:
                sleep(5)
                print('Test failed!')
                
                

if __name__ == '__main__':
    duration = 60
    n_scenarios = 1000
    n_runs = 10
    mht = MHFlakyTester(duration, n_scenarios, n_runs)
    ##  Comment below if you want to resume previous tests...
    mht.saveTestConfigs()
    # mht.run()