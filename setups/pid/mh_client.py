import random
import carla

##  Connecting to Carla server...
client = carla.Client('localhost', 2000)
##  Loading Carla world with a specific map...
client.load_world('Town05')
world = client.get_world()
##  Specifying weather conditions...
weather = world.get_weather()
weather.cloudiness = 0.
weather.percipitaion = 0.
weather.fog_density = 0.
world.set_weather(weather)

##  Vehicle...
##  Get the blueprint library and filter for the vehicle blueprints
vehicle_bps = world.get_blueprint_library().filter('*vehicle*')

##  Randomly choose a vehicle blueprint to spawn
vehicle_bp = random.choice(vehicle_bps)

##  We need a place to spawn the vehicle that will work so we will
##  use the predefined spawn points for the map and randomly select one
# spawn_point = random.choice(world.get_map().get_spawn_points())

##  Retrieve the spectator object
spectator = world.get_spectator()

##  Get the location and rotation of the spectator through its transform
transform = spectator.get_transform()
spawn_point = spectator.get_transform()


location = transform.location
rotation = transform.rotation

# Now let's spawn the vehicle
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

