import json
import rocketpy  as rk

def environment_init(variables_filepath):
    environment = rk.Environment()

    with open(variables_filepath, 'r', encoding='utf-8') as variables_file:
        variables_data = json.load(variables_file)

    return environment

#main block
if __name__ == '__main__':
    variables_filepath = "variables.json"
    env = environment_init(variables_filepath)
