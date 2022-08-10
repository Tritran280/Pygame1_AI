
import pygame
import neat
import os
import random

pygame.init()
screen=pygame.display.set_mode((800,600))
clock = pygame.time.Clock()

class Ball:
	def __init__(self):
		self.x = 500
		self.y = 100
		self.x_speed = random.randint(4, 6)
		self.y_speed = random.randint(4, 5)
		self.x_speed_scale = 1
		self.y_speed_scale = 1
		self.surface = pygame.transform.scale(pygame.image.load("./Img/imgball.png"), (20, 20))
		self.hit_bottom = False
	
	def move(self):
		self.x += self.x_speed * self.x_speed_scale
		self.y += self.y_speed * self.y_speed_scale
	
	def set_x_speed_scale(self, value):
		self.x_speed_scale = value
	def set_y_speed_scale(self, value):
		self.y_speed_scale = value

class Bar:
	def __init__(self, speed) -> None:
		self.x = 500
		self.y = 500
		self.speed = speed
		self.surface = pygame.transform.scale(pygame.image.load("./Img/a.png"), (100, 5))

	def move(self, distance):
		self.x += self.speed * distance

class Draw_:
	def __init__(self, screen):
		self.screen = screen
		self.cotngang = pygame.transform.scale(pygame.image.load("./Img/a.png"), (800, 10))
		self.cotdung = pygame.transform.scale(pygame.image.load("./Img/a.png"), (10, 600))
		self.RED=(255,0,0)

	def draw_ball(self, ball):
		return self.screen.blit(ball.surface, (ball.x, ball.y))
	def draw_bar(self, bar):
		return self.screen.blit(bar.surface, (bar.x, bar.y))
	def draw_border_top(self):
		return self.screen.blit(self.cotngang, (0, 0))
	def draw_border_bottom(self):
		return self.screen.blit(self.cotngang, (0, 590))
	def draw_border_left(self):
		return self.screen.blit(self.cotdung, (0, 0))
	def draw_border_right(self):
		return self.screen.blit(self.cotdung, (790, 0))

drawer = Draw_(screen)

def run(genomeTuples, config):
	global drawer
	genomes = []
	networks = []
	balls = []
	bars = []

	for genome_id, genome in genomeTuples:
		genome.fitness = 0
		network = neat.nn.FeedForwardNetwork.create(genome, config)

		genomes.append(genome)
		networks.append(network)
		balls.append(Ball())
		bars.append(Bar(7))

	while True:
		screen.fill((0, 0, 0))
		clock.tick(60)

		for event in pygame.event.get():
			if event.type==pygame.QUIT:
				pygame.display.update()
				pygame.quit()

		if len(balls) <= 0:
			break

		top_hitbox = drawer.draw_border_top()
		bottom_hitbox = drawer.draw_border_bottom()
		left_hitbox = drawer.draw_border_left()
		right_hitbox = drawer.draw_border_right()

		for i, _ in enumerate(balls):
			genome = genomes[i]
			network = networks[i]
			ball = balls[i]
			bar = bars[i]
			ball.move()

			output = network.activate((bar.x, ball.x))

			if output[0] < -0.3:
				bar.move(-1)
			else:
				bar.move(1)

			ball_hitbox = drawer.draw_ball(ball)
			bar_hitbox = drawer.draw_bar(bar)

			if ball_hitbox.colliderect(right_hitbox):
				ball.set_x_speed_scale(-1)
			
			if ball_hitbox.colliderect(left_hitbox):
				ball.set_x_speed_scale(1)

			if ball_hitbox.colliderect(top_hitbox):
				ball.set_y_speed_scale(1)

			if ball_hitbox.colliderect(bottom_hitbox):
				ball.hit_bottom = True

			if ball_hitbox.colliderect(bar_hitbox):
				ball.set_y_speed_scale(-1)
				genome.fitness += 1
		
		for i, ball in enumerate(balls):
			if ball.hit_bottom:
				genomes[i].fitness -= abs(bars[i].x - ball.x)
				genomes.pop(i)
				networks.pop(i)
				balls.pop(i)
				bars.pop(i)

		pygame.display.update()
		
if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config.txt")

	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

	p = neat.Population(config)
	p.add_reporter(neat.StdOutReporter(True))
	p.add_reporter(neat.StatisticsReporter())
	
	winner = p.run(run, 50)