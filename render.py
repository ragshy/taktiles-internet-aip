import pygame

pygame.init()

BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 0, 255, 0)
RED = ( 255, 0, 0)

width = 700
height = 700
size = (width, height)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Position Render")

clock = pygame.time.Clock()

loop = True
while loop:  
  for event in pygame.event.get(): # User did something
    if event.type == pygame.QUIT: # If user clicked close
      loop = False # Flag that we are done so we can exit the while loop
 
  # --- Logic should go here
  
  # --- Drawing code should go here
  screen.fill(WHITE)
  pygame.draw.rect(screen, RED, [0, height/2, 50, 50],0)
  pygame.draw.ellipse(screen, BLACK, [20,20,250,100], 2)

  # --- Go ahead and update the screen with what we've drawn.
  pygame.display.update()
  clock.tick(60)


pygame.quit()