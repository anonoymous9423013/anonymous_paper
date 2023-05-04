import random
import carla
from time import sleep, time
from datetime import datetime
import logging as log


class MHConfig:
    def __init__(self, map='Town03', ego_vehicle_bp=None, vehicle_bp=None, filename=None, ego_vehicle_bp_index=0, vehicle_bp_index=1):
        self.now = datetime.now()
        self.dt_string = self.now.strftime("%d_%m_%Y_%H_%M_%S")
        if filename:
            self.filename = filename
        else:
            self.filename = self.dt_string
        
        ##  Connecting to Carla server...
        self.client = carla.Client('localhost', 2000)
        ##  Loading Carla world with a specific map...
        self.client.load_world(map)
        self.world = self.client.get_world()
        self.is_running = False
        
        self.vehicles = []
        self.pedestrians = []
        self.cameras = []

        ##  Specifying vehicles...
        self.ego_spawn_point = 110
        self.fronts_spawn_points = [107, 108, 28, 29]
        self.backs_spawn_points = [109, 31, 30]
        self.opposites_spawn_points = [27, 26, 35]

        ##  Specifying routes...
        self.route1_indices = [107, 28, 155]
        self.route1_indices = [157, 202, 168, 155, 173, 183]
        self.route1_indices = [157, 202, 168]
        self.route1 = [self.getSpawnPoints()[i].location for i in self.route1_indices]

        ##  Ego vehicle...
        if ego_vehicle_bp is None:
            self.ego_vehicle_bp = self.world.get_blueprint_library().filter('vehicle.*')[ego_vehicle_bp_index]
        else:
            self.ego_vehicle_bp = ego_vehicle_bp
        self.ego_vehicle = self.addVehicle(self.getSpawnPoints()[self.ego_spawn_point], self.ego_vehicle_bp, attach_camera=True, collision_sensor=True, lane_invasion_sensor=True, obstacle_sensor=False)

        if vehicle_bp is None:
            self.vehicle_bp = self.world.get_blueprint_library().filter('vehicle.*')[vehicle_bp_index]
        else:
            self.vehicle_bp = vehicle_bp

        

    def setWeather(self, cloudiness, percipitaion, fog_density):
        ##  Specifying weather conditions...
        weather = self.world.get_weather()
        weather.cloudiness = cloudiness
        weather.percipitaion = percipitaion
        weather.fog_density = fog_density
        self.world.set_weather(weather)

    def addVehicle(self, spawn_point, vehicle_bp=None, attach_camera=False, collision_sensor=False, lane_invasion_sensor=False, obstacle_sensor=False):
        ##  Now let's spawn the vehicle
        if vehicle_bp is None:
            vehicle_bp = self.vehicle_bp
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            self.vehicles.append(vehicle)
        if attach_camera:
            ##  Attach a camera to the vehicle
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '110')
            camera_transform = carla.Transform(carla.Location(x=-10, z=4))
            camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
            spectator = self.world.get_spectator()
            spectator.set_transform(camera.get_transform())
            self.cameras.append(camera)
            ##  Record the camera with date and time in folder name...
            camera.listen(lambda image: image.save_to_disk('output/%s/%s.png' % (self.filename, image.frame)))
            if __name__ == '__main__':
                print('Camera attached to vehicle')

        if collision_sensor:
            ##  Attach a collision sensor to the vehicle
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            collision_transform = carla.Transform(carla.Location(x=0, z=0))
            self.collision_sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
            self.collision_sensor.listen(lambda event: log.info('Collision detected: %s' % event))
            if __name__ == '__main__':
                print('Collision sensor attached to vehicle')
        if lane_invasion_sensor:
            ##  Attach a lane invasion sensor to the vehicle
            lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            lane_invasion_transform = carla.Transform(carla.Location(x=0, z=0))
            self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, lane_invasion_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
            self.lane_invasion_sensor.listen(lambda event: log.info('Lane invasion detected: %s' % event))
            if __name__ == '__main__':
                print('Lane invasion sensor attached to vehicle')
        if obstacle_sensor:
            ##  Attach an obstacle sensor to the vehicle
            obstacle_bp = self.world.get_blueprint_library().find('sensor.other.obstacle')
            obstacle_transform = carla.Transform(carla.Location(x=0, z=0))
            self.obstacle_sensor = self.world.spawn_actor(obstacle_bp, obstacle_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
            self.obstacle_sensor.listen(lambda event: log.info('Obstacle detected: %s' % event))
            if __name__ == '__main__':
                print('Obstacle sensor attached to vehicle')

        return vehicle

    def get_distance_from_center_lane(self):
        ego_vehicle_location = self.ego_vehicle.get_location()

        waypoint = self.world.get_map().get_waypoint(ego_vehicle_location, project_to_road=True)

        ego_vehicle_loc = carla.Location(x=ego_vehicle_location.x, y=ego_vehicle_location.y, z=0.0)
        log.info("Distance From center of Lane: " + str(ego_vehicle_loc.distance(waypoint.transform.location)))
        return ego_vehicle_loc.distance(waypoint.transform.location)
    
    def get_distance_from_destination(self):
        ego_vehicle_location = self.ego_vehicle.get_location()
        self.final_destination = carla.Location(x=self.route1[-1].x, y=self.route1[-1].y, z=0.0)
        dist = ego_vehicle_location.distance(self.final_destination)-10.8; #9.8 for v1
        log.info("Distance from Final Destination: " + str(dist))
        return dist

        
    def addPedestrian(self, spawn_point, pedestrian_bp):
        ##  Now let's spawn the pedestrian...
        pedestrian = self.world.try_spawn_actor(pedestrian_bp, spawn_point)
        self.pedestrians.append(pedestrian)
        return pedestrian
    
    def getSpawnPoints(self):
        spawn_points = self.world.get_map().get_spawn_points()
        return spawn_points

    def drawSpawnPoints(self):
        spawn_points = self.getSpawnPoints()
        for i, spawn_point in enumerate(spawn_points):
            self.world.debug.draw_string(spawn_point.location, str(i), draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)            
            

    def getVehicleBlueprints(self):
        ##  Get the blueprint library and filter for the vehicle blueprints
        vehicle_bps = self.world.get_blueprint_library().filter('*vehicle*')
        return vehicle_bps
    
    def getPedestrianBlueprints(self):
        ##  Get the blueprint library and filter for the pedestrian blueprints
        pedestrian_bps = self.world.get_blueprint_library().filter('*walker*')
        return pedestrian_bps

    def addRelativeVehicle(self, vehicle_bp=None, mode='front', n=1, random_bp=False):
        if mode == 'front':
            for i in range(n):
                if random_bp:
                    vehicle_bp = random.choice(self.getVehicleBlueprints())
                self.addVehicle(self.getSpawnPoints()[self.fronts_spawn_points[i]], vehicle_bp)
        elif mode == 'back':
            for i in range(n):
                if random_bp:
                    vehicle_bp = random.choice(self.getVehicleBlueprints())
                self.addVehicle(self.getSpawnPoints()[self.backs_spawn_points[i]], vehicle_bp)
        elif mode == 'opposite':
            for i in range(n):
                if random_bp:
                    vehicle_bp = random.choice(self.getVehicleBlueprints())
                self.addVehicle(self.getSpawnPoints()[self.opposites_spawn_points[i]], vehicle_bp)


        
    def updateSpectator(self):
        camera = self.cameras[0]
        spectator = self.world.get_spectator()
        spectator.set_transform(camera.get_transform())

    def startTrafficManager(self):
        self.tm = self.client.get_trafficmanager(8000)
        tm_port = self.tm.get_port()
        self.tm.set_synchronous_mode(True)
        if __name__ == '__main__':
            print('Traffic Manager running on port: %d' % tm_port)
        for v in self.vehicles:
            v.set_autopilot(True, tm_port)

        ##  For dangerous vehicle set p=100...
        p = 0
        self.tm.ignore_lights_percentage(self.ego_vehicle, p)
        self.tm.random_left_lanechange_percentage(self.ego_vehicle, p)
        self.tm.random_right_lanechange_percentage(self.ego_vehicle, p)
        if p == 100:
            self.tm.distance_to_leading_vehicle(self.ego_vehicle, 0.0)
            self.tm.vehicle_percentage_speed_difference(self.ego_vehicle, 0.0)
        ##  Attach the spectator to the first vehicle...
        spectator = self.world.get_spectator()
        spectator.set_transform(self.vehicles[0].get_transform())
        ##  Set route for the ego vehicle...
        self.tm.set_path(self.ego_vehicle, self.route1)
        if __name__ == '__main__':
            print('Ego vehicle route set')
            print('self.ego_vehicle: ', self.ego_vehicle)
            print('self.route1: ', self.route1)


    def run(self, time_step=0.5, time_duration=60):
        fileh = log.FileHandler(f'output/log_{self.filename}.txt', 'a')
        fileh.setFormatter(log.Formatter('%(asctime)s %(message)s'))
        for hdlr in log.getLogger().handlers[:]:  # remove all old handlers
            log.getLogger().removeHandler(hdlr)
        log.getLogger().addHandler(fileh)
        log.getLogger().setLevel(log.DEBUG)
        log.debug('Starting logging...')

        t0 = time()
        self.is_running = True
        ##  Run the simulation
        while True:
            sleep(time_step)
            self.updateSpectator()
            self.world.tick()
            self.get_distance_from_center_lane()
            self.get_distance_from_destination()
            if time() - t0 >= time_duration:
                self.is_running = False
                break
    
        

if __name__ == '__main__':
    mhc = MHConfig()
    mhc.setWeather(0, 0, 0)
    spawn_points = mhc.getSpawnPoints()
    vehicle_bps = mhc.getVehicleBlueprints()
    vehicle_bp = vehicle_bps.filter('vehicle.audi.*')[1]
    mhc.addRelativeVehicle(vehicle_bp, mode='front', n=3, random_bp=False)
    mhc.addRelativeVehicle(vehicle_bp, mode='back', n=2, random_bp=False)
    mhc.addRelativeVehicle(vehicle_bp, mode='opposite', n=3, random_bp=False)
    mhc.addPedestrian(mhc.getSpawnPoints()[32], mhc.getPedestrianBlueprints()[0])
    print('No of vehicles: ', len(mhc.vehicles))
    mhc.drawSpawnPoints()
    mhc.startTrafficManager()
    mhc.run(0.05, 60)
