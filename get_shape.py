import pickle
data = pickle.load(open('checkpoints/checkpoint_000975/checkpoint-975', 'rb'))
worker = pickle.loads(data['worker'])
print([k for k in worker['state'].keys()])
