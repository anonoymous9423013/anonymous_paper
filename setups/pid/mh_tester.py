import os
import sys
import multiprocessing as mp
from time import sleep

##  status[0] ->  True if Carla server is running

def runCarlaServer(status):
    command = 'sudo docker run --privileged --rm --gpus all --net=host --name mhsim -e DISPLAY=$DISPLAY carlasim/carla:latest /bin/bash ./CarlaUE4.sh'
    os.system(command)
    for i in range(10):
        print('*' * 100)
    status.put(False)
    print('Process done!!!')

def findLatestTest(n_runs=10):
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
        if test_index < n_runs:
            return i, test_index

    ##  When all tests are done...
    return -1, -1

def killCarlaServer():
    os.system('sudo docker stop mhsim')
    os.system('sudo docker rm mhsim')

def runNextTest(n_runs=10, duration=15):
    scenario_index, test_index = findLatestTest(n_runs)
    print(f'Scenario index: {scenario_index}, Test index: {test_index}')
    
    if scenario_index == -1:
        print('All tests done!!!')
        return -1, -1

    with open('mh_flaky_test_1.txt', 'r') as f:
        scenario = f.readlines()[scenario_index].strip()

    command = f'python3 mh_single_test.py {scenario.replace("," ," ")} {test_index} {duration}'
    print(f'Command: {command}')
    res = os.system(command)
    return scenario_index, test_index, res
    

    

class MHTester:
    def __init__(self) -> None:
        pass 

if __name__ == '__main__':
    q = mp.Queue()
    p = mp.Process(target=runCarlaServer, args=(q,), name='Carla Server', daemon=False)
    p.start()
    print('Loading Carla server...')
    sleep(15)
    while True:
        if not q.empty():
            print('Carla server is not running...')
            q = mp.Queue()
            p = mp.Process(target=runCarlaServer, args=(q,), name='Carla Server', daemon=False)
            p.start()
            sleep(15)

        print('Going for the next test...')
        sleep(5)
        try:
            scenario_index, test_index, res = runNextTest(10, 60)
            print(f'Result: {res}')
            if scenario_index == -1:
                break
            if res:
                p.kill()
                q.put(False)
                killCarlaServer()
        except:
            p.kill()
            q.put(False)
            killCarlaServer()
        