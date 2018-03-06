


class Particle(object):
    """
        PSO particle
    """

    def __init__(self, pos, vel):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.pos = pos
        self.vel = vel
        self.bestPos = self.pos


class Swarm(object):
    """
        Swarm containing PSO particles
    """

    def __init__(self, nParticles, w):
        self.nParticles = nParticles
        self.inertia = w
        self.particles = []
        initParticles()

    # create the particles
    def initParticles(self):
        for x in xrange(0, self.nParticles):
            p = Particle()
            self.particles.append(p)

    def updateParticles(self):
        for p in self.particles:
            p.vel = self.inet


# number of particles, inertia, and acceleration constants
particles = 500
w = 1
c1 = 1
c2 = 1
swarm = Swarm(particles, inertia, acc_c1, acc_c2)
