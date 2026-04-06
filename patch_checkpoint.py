import pickle

with open('checkpoint_098/checkpoint-98', 'rb') as f:
    data = pickle.load(f)

worker = pickle.loads(data['worker'])
if 'default_policy' in worker['state']:
    policy_state = worker['state']['default_policy']
    if 'weights' not in policy_state:
        # Filter out _optimizer_variables since they shouldn't be in weights
        weights_dict = {k: v for k, v in policy_state.items() if k != '_optimizer_variables'}
        worker['state']['default_policy'] = {
            'weights': weights_dict,
            'global_timestep': 0
        }
        data['worker'] = pickle.dumps(worker)
        with open('checkpoint_098/checkpoint-98-patched', 'wb') as f:
            pickle.dump(data, f)
        print("Checkpoint patched and saved as checkpoint-98-patched")
    else:
        print("Checkpoint already has 'weights' key.")
else:
    print("Cannot find 'default_policy' in worker state.")
