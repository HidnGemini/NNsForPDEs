import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.animation as animation

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# ---- Simple RNN Model ----
class HeatRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x, steps):
        batch_size, nx = x.shape
        h = torch.zeros(1, batch_size, self.gru.hidden_size, device=x.device)
        preds = []
        for _ in range(steps):
            x_input = x.unsqueeze(1)  # shape: [batch, 1, nx]
            out, h = self.gru(x_input, h)
            x = self.decoder(out.squeeze(1))
            preds.append(x)
        return torch.stack(preds)  # shape: [steps, batch, nx]

nx = 50
x = torch.linspace(0, 1, nx).unsqueeze(0)  # shape: [1, nx]
u0 = torch.sin(torch.pi * x)  # initial temperature profile

model_file = "firstPeRCNN.pth"
model = torch.load(model_file, weights_only=False, map_location=torch.device("cpu"))

n_t = 1000
u_pred = model(u0, n_t)

# plot anim

fig, ax = plt.subplots()

pred, = ax.plot(torch.Tensor.cpu(u_pred[0, 0].detach()), '--', label="Predicted")
# real, = ax.plot(torch.Tensor.cpu(u_true[0, 0].detach()), label="True")

def animate(step):
    t = 10*step / n_t
    ax.set_title(f"Time t={t}")
    # u_pred = model(u0, step)
    pred.set_ydata(torch.Tensor.cpu(u_pred[step, 0].detach()))
    # real.set_ydata(torch.Tensor.cpu(u_true[0, step].detach()))
    return pred,

ani = animation.FuncAnimation(
    fig, animate, interval=10, blit=False, save_count=50, frames=n_t)

plt.show()

ani.save("u_animation.gif", writer=animation.PillowWriter(fps=20))
