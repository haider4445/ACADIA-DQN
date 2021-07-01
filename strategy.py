import random

class Strategy:

	def __init__(self, Totalsteps):
		self.Totalsteps = Totalsteps

	def randomStrategy(self):
		if random.randint(0,100) % 2 == 0:
			return False
		else:
			return True
	def leastStepsStrategy(self):
		pass
