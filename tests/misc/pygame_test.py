import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np

import pygame

import time

fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
line = ax.plot(range(1, 10), range(1, 10))

canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.buffer_rgba()

pygame.init()

window = pygame.display.set_mode(raw_data.shape[:2])
screen = pygame.display.get_surface()

size = canvas.get_width_height()

surf = pygame.image.frombuffer(raw_data, size, "RGBA")
screen.blit(surf, (0,0))
pygame.display.flip()

crashed = False
start_time = time.time()
clock = pygame.time.Clock()
while not crashed:
    if time.time() - start_time > 2:
        line = line.pop(0)
        line.remove()
        line = ax.plot(range(1, 10), 10*np.random.rand(9), c='k')
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        surf = pygame.image.frombuffer(raw_data, size, "RGBA")
        screen.blit(surf, (0,0))
        pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
    clock.tick(1)