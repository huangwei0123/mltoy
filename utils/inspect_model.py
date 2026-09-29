import torch

ckpt = torch.load(
    "models/t2m_cpu.pt",
    map_location="cpu",
    weights_only=False
)

state = ckpt["model_state_dict"]

for k, v in state.items():
    print(k, v.shape)
