from world import BoxWorld

def create_world():
    # Define world with obstacles
    # Outer dimensions
    world = BoxWorld([[0, 10], [0, 10], [0, 10]])
    
    # in order: x,y,z,lenght,width,height
    world.add_box(0, 1, 0, 4, 2, 3)
    world.add_box(0, 6, 0, 4, 1, 1)
    world.add_box(4, 1, 0, 4, 1, 1)
    world.add_box(7, 7, 0, 3, 3, 1)

    return world
