from mh_config import MHConfig
import sys



if __name__ == '__main__':
    scenario = sys.argv[1:]
    weather_params = [int(i) for i in scenario[:3]]
    ego_vehicle_bp_index = int(scenario[3])
    vehicle_bp_index = int(scenario[4])
    n_fronts = int(scenario[5])
    n_backs = int(scenario[6])
    n_opposites = int(scenario[7])
    test_index = int(scenario[8])
    test_duration = int(scenario[9])

    mhc = MHConfig(ego_vehicle_bp_index=ego_vehicle_bp_index, vehicle_bp_index=vehicle_bp_index, filename=','.join(scenario[:-2])+f'_{test_index}')

    mhc.setWeather(*weather_params)
    mhc.addRelativeVehicle(None, mode='front', n=n_fronts, random_bp=False)
    mhc.addRelativeVehicle(None, mode='back', n=n_backs, random_bp=False)
    mhc.addRelativeVehicle(None, mode='opposite', n=n_opposites, random_bp=False)
    mhc.startTrafficManager()
    mhc.run(0.05, test_duration)
    with open('mh_flaky_test_1_done.txt', 'a') as f:
        f.write(','.join([str(i) for i in scenario[:-2]]) + f'_{test_index}' + '\n')
    print('Test finished!')
    