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