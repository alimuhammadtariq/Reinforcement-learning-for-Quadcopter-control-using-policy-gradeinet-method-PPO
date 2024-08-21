import pygame
from PIL import Image, ImageSequence

def calculate_next_state(current_state, action, delta_t=0.1):

    start_y, start_v = current_state
    # next y = start_y + V_new*deta_t; v_new = v_old + a_net * delta_t

    new_v = start_v + (action - 9.8) * delta_t
    new_y = start_y + new_v * delta_t

    if new_y > 600 or new_y < 90:
        return 90, 0
    else:
        return new_y, new_v

# initialing the graphics window


pygame.init()
image_path = "Quadcopter.gif"
width, screen_height = 800, 600
screen = pygame.display.set_mode((width, screen_height))
pygame.display.set_caption("Quadcopter Physics Engine")
frame_duration = 100

# loading quadcopter Image gif and resizing

resize_dimensions_quadcopter = (80, 80)
quadcopter = Image.open(image_path)

# iterate for all gif images and downscale the dimensions and convert them to pygame surface
frames = [frame.copy().resize(resize_dimensions_quadcopter, Image.LANCZOS) for frame in ImageSequence.Iterator(quadcopter)]
pygame_frames = [pygame.image.fromstring(frame.tobytes(), frame.size, frame.mode) for frame in frames]


# Animation index
frame_index = 0


# Initial state
current_state = [300, 0]  # height, y_velocity
delta_t = 0.1
# acc_increment = 10
# acc_y = 0


# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    action = 0

    if keys[pygame.K_UP]:
        action = 30

    if keys[pygame.K_DOWN]:
        action = -10

    current_state = calculate_next_state(current_state,action, delta_t)
    print("velocity", current_state[1], "y=", current_state[0])


    #if keys[pygame.K_LEFT]:
    #    acc_y = -50
    #    current_state = calculate_next_state(current_state, acc_y, delta_t)
    #
    # if event.type == pygame.KEYUP:
    #     action=0
    #     current_state = calculate_next_state(current_state, action, delta_t)


    # Update the frame index
    frame_index = (frame_index + 1) % len(pygame_frames)

    # Clear the screen
    screen.fill((255, 255, 255))  # Fill with white color
    # print("acc", acc_y*-1,"state",  current_state)
    # Draw the current frame at position (x, y)
    screen.blit(pygame_frames[frame_index], tuple([400,screen_height-current_state[0]]))

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    pygame.time.wait(frame_duration)

# Quit Pygame
pygame.quit()



