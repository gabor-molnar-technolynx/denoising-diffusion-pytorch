import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import coremltools as ct

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

training_images = torch.rand(1, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()
# after a lot of training



dummy_img = torch.rand((1,3,128,128)).float()
# dummy_timestep = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
dummy_timestep = torch.randint(0, 100, (1,)).long()
model_ts = torch.jit.trace(diffusion.model, (dummy_img, dummy_timestep))
model_ct = ct.convert(model_ts,
                              inputs=[ct.TensorType(name="img_input", shape=dummy_img.shape),
                                      ct.TensorType(name="timestep_input", shape=dummy_timestep.shape)],
                              outputs=[
                                  ct.TensorType(name="noise_prediction")])


sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)

print("done")