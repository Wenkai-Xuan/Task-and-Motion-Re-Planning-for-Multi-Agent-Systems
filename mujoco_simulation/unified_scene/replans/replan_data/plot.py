import matplotlib.pyplot as plt
import os
import json
import pandas as pd

def latest_sequences(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()] #full paths for all subdirectories within folder_path
    # print(subfolders)
    sequences = [s for s in subfolders if s.startswith(f"{folder_path}/sequence_generation")]  # the list of all generated sequences
    plans = [s for s in subfolders if s.startswith(f"{folder_path}/sequence_plan")]
    # print(sequences)
    if not plans: # "plan_for_sequence" failed
        return None
    
    latest_sequences = max(sequences, key=os.path.getmtime)
    with open(f'{latest_sequences}/sequences.json', 'r') as file:
        seq = json.load(file)
    
    # "generate_candidate_sequences" failed may cause "plan_for_sequence" success
    if all(list(tasks.values())[0] is None for tasks in seq['sequences']): 
        return None
    
    latest_plans = max(plans, key=os.path.getmtime)

    return latest_plans


path = './replan_ini_uni'
train = pd.read_parquet(path + '/uni_train.parquet')
test = pd.read_parquet(path + '/uni_test.parquet')
scenes = ['conveyor_5_rela_15', 'husky_25_rela_25', 'random_520_rela_25', 'shelf_52_rela_25']
dirs = ['replan_ini_conveyor_5_20250302_174502', 'replan_ini_husky_25_20250308_224336', 
         'replan_ini_random_520_20250308_225030', 'replan_ini_shelf_52_20250317_143018']
colors = ['blue', 'orange', 'green', 'red']

# # plot the histogram of the makespan of replans for each scene
# for elem in scenes:
#     freq = []
#     data = pd.read_parquet(path + '/' + elem + '.parquet')
#     for i in range(len(data)):
#         freq.append(data['metadata'][i]['metadata']['makespan'])

#     # random old plan length: (186+1)
#     freq_random = [x + 0 for x in freq]

#     env = elem.split('_')[0]
#     # print(freq)
#     plt.hist(freq_random, bins=15, color=colors[scenes.index(elem)], edgecolor='white', alpha=0.7)

#     plt.title(f'Makespan of plans for {env} scene')
#     plt.xlabel('Makespan (Number of steps)')
#     plt.ylabel('Number of plans')

#     plt.show()

# plot the histogram of the makespan of full plans for each scene
for dir in dirs:
    # print(dir)
    freq = []
    path = './' + dir
    for replan_config in sorted(os.listdir(path)):
        replan_config_path = os.path.join(path, replan_config)
        
        # only when replan_config is a directory named as drop-step, let's proceed and iterate later
        if os.path.isdir(replan_config_path) and replan_config.isdigit():
            old_makespan = int(replan_config)
            plan_directory = latest_sequences(replan_config_path) # get the latest sequence_plan directory
            if plan_directory is None: # skip the steps that "plan_for_sequence" failed
                continue

            for single_plan in sorted(os.listdir(plan_directory)): #plan_directory is definitely a directory
            
                single_plan_path = os.path.join(plan_directory, single_plan)
                # print(single_plan)

                if os.path.isdir(single_plan_path):
                    # print(single_plan_path)
                    for file in os.listdir(single_plan_path):
                        file_path = os.path.join(single_plan_path, file)
                        # print(file_path)
                        if os.path.isfile(file_path):
                            if file.startswith('metadata'):
                                with open(file_path, 'r') as f:
                                    metadata = json.load(f)
                                    freq.append(metadata['metadata']['makespan'] + old_makespan)


    env = dir.split('_')[2]
    # print(freq)
    plt.hist(freq, bins=15, color=colors[dirs.index(dir)], edgecolor='white', alpha=0.7)

    plt.title(f'Full makespan of plans for {env} scene')
    plt.xlabel('Makespan (Number of steps)')
    plt.ylabel('Number of plans')
    plt.show()