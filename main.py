import torch
import numpy as np
import matplotlib.pyplot as plt

n_normal = 50
normal_x = np.random.rand(n_normal)
noise = np.random.normal(loc=0., scale=0.05, size=n_normal)
normal_y = 2.1 * normal_x + 0.5 + noise

n_anomaly = 10
anomaly_x = np.random.rand(n_anomaly)
anomaly_y = np.random.rand(n_anomaly)

x = np.concatenate((normal_x, anomaly_x))
y = np.concatenate((normal_y, anomaly_y))


class Regressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.a * x + self.b


def l2_loss(y, y_pred):
    """Mean Squared Loss."""
    return (y - y_pred).pow(2).mean()


def huber_loss(y, y_pred, sigma=0.1):
    r = (y - y_pred).abs()
    loss = (r[r <= sigma]).pow(2).mean()
    loss += (r[r > sigma]).mean() * sigma - sigma**2/2
    return loss


m = Regressor()
optimizer = torch.optim.Adam(m.parameters(), lr=1e-1)


def train(x, y, loss_fn, n_step=200):
    for step in range(n_step):
        x_ = torch.Tensor(x)#.unsqueeze(dim=0)
        y_ = torch.Tensor(y)#.unsqueeze(dim=0)

        y_pred = m(x_)
        optimizer.zero_grad()
        loss = loss_fn(y_, y_pred)
        loss.backward()
        optimizer.step()

    return m.a.data.item(), m.b.data.item()

plt.cla()
plt.scatter(normal_x, normal_y)
plt.scatter(anomaly_x, anomaly_y)
x_lin = np.arange(0, 3)


a, b = train(normal_x, normal_y, l2_loss)
plt.plot(x_lin, a*x_lin+b, label='l2-normal')
print('normal a, b: ', a, b)

a, b = train(x, y, l2_loss)
plt.plot(x_lin, a*x_lin+b, label='l2-normal & anomaly')
print('normal-anomaly a, b: ', a, b)

a, b = train(x, y, huber_loss)
plt.plot(x_lin, a*x_lin+b, label='huber(sigma=0.1)-normal & anomaly')
print('huber normal-anomaly a, b: ', a, b)
plt.legend()
plt.show()









