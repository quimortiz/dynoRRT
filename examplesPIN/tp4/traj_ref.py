import numpy as np


class TrajRef:
    def __init__(self, q0, omega, amplitude):
        self.q0 = q0.copy()
        self.omega = omega
        self.amplitude = amplitude
        self.q = self.q0.copy()
        self.vq = self.q0 * 0
        self.aq = self.q0 * 0

    def position(self, t):
        """Compute a reference position for time <t>."""
        self.q.flat[:] = self.q0
        self.q.flat[:] += self.amplitude * np.sin(self.omega * t)
        return self.q

    def velocity(self, t):
        """Compute and return the reference velocity at time <t>."""
        self.vq.flat[:] = self.omega * self.amplitude * np.cos(self.omega * t)
        return self.vq

    def acceleration(self, t):
        """Compute and return the reference acceleration at time <t>."""
        self.aq.flat[:] = -self.omega**2 * self.amplitude * np.sin(self.omega * t)
        return self.aq

    def __call__(self, t):
        return self.position(t)

    def copy(self):
        return self.q.copy()


if __name__ == "__main__":
    # %jupyter_snippet main
    qdes = TrajRef(np.array([0, 0, 0.0]), omega=np.array([1, 2, 3.0]), amplitude=1.5)
    t = 0.2
    print(qdes(t), qdes.velocity(t), qdes.acceleration(t))
    # %end_jupyter_snippet
