import soccer_twos

# Create a random env 
env = soccer_twos.make(single_player=True)

# Start the game
obs = env.reset()

print("OBSERVATION TYPE:", type(obs))
print("OBSERVATION SHAPE:", obs.shape)
print("OBSERVATION DATA:", obs)
