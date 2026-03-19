import torch
from matd3_agent import Actor

# Create a dummy actor with your dimensions
model = Actor(state_dim=24, action_dim=4, max_action=1.0)
torch.save(model.state_dict(), "trained_matd3_actor.pth")
print("Placeholder model created for testing.")