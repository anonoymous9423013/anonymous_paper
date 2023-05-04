import os 

class MHTester:
    def __init__(self, n_repeats=10, config_file='config.txt', output_folder='output', output_file='output.txt'):
        self.n_repeats = n_repeats
        self.config_file = config_file
        self.output_folder = output_folder
        self.output_file = output_file

    def findNextConfig(self):
        with open(self.config_file, 'r') as f:
            configs = f.readlines()
        configs = [x.strip().rstrip() for x in configs]
        configs.sort()
        if not os.path.isfile(self.output_file):
            with open(self.output_file, 'w') as f:
                pass 
        with open(self.output_file, 'r') as f:
            outputs = f.readlines()
        if not len(outputs):
            return configs[0]
        latest_repeat = int(outputs[-1].split('_')[1])
        current_config_index = configs.index(outputs[-1].split('_')[0])
        if latest_repeat >= self.n_repeats - 1:
            return configs[current_config_index + 1]
        return configs[current_config_index]
    
    def findNoOfRepeats(self, config_str):
        with open(self.output_file, 'r') as f:
            outputs = f.readlines()
        outputs = [x.strip().rstrip() for x in outputs]
        outputs = [x.split('_')[0] for x in outputs]
        return outputs.count(config_str)

                    


if __name__ == '__main__':
    tester = MHTester(n_repeats=10)
    next_config = tester.findNextConfig()
    print('Next config:', next_config)
    print('No of repeats:', tester.findNoOfRepeats(next_config))