import random
import numpy as np

# Module-level cache
_cached_red_fields = None
_cached_green_fields = None

def randomize_obstacles(supervisor, train=False, seed=None):
    """
    Randomly places obstacles on the track in West, North, and East sections.
    If train is True, it caches the Webots Field objects in module-level variables
    to optimize training reset times.
    The seed parameter can be an integer seed or a numpy Generator (e.g. self.np_random).
    """
    global _cached_red_fields, _cached_green_fields
    
    # 1. Load obstacle translation fields (from cache or lookup)
    if train:
        if _cached_red_fields is None or _cached_green_fields is None:
            _cached_red_fields = []
            _cached_green_fields = []
            for i in range(4):
                node_red = supervisor.getFromDef(f"OBSTACLE_RED_{i}")
                if node_red:
                    _cached_red_fields.append(node_red.getField("translation"))
                node_green = supervisor.getFromDef(f"OBSTACLE_GREEN_{i}")
                if node_green:
                    _cached_green_fields.append(node_green.getField("translation"))
        red_fields = _cached_red_fields
        green_fields = _cached_green_fields
    else:
        # Non-cached lookup for evaluation/single-run start
        red_fields = []
        green_fields = []
        for i in range(4):
            node_red = supervisor.getFromDef(f"OBSTACLE_RED_{i}")
            if node_red:
                red_fields.append(node_red.getField("translation"))
            node_green = supervisor.getFromDef(f"OBSTACLE_GREEN_{i}")
            if node_green:
                green_fields.append(node_green.getField("translation"))

    # Reset all obstacles to default/off-track positions first (e.g. [0, 0, 0.05])
    for field in red_fields + green_fields:
        field.setSFVec3f([0.0, 0.0, 0.05])
        
    if not red_fields and not green_fields:
        print("[obstacle_randomizer] Warning: No red or green obstacles found in the world.")
        return

    # 2. Determine random generator/seed
    if isinstance(seed, (np.random.Generator, np.random.RandomState)):
        rng = seed
    elif isinstance(seed, int):
        rng = np.random.default_rng(seed)
        random.seed(seed)
    else:
        rng = None

    def choose(options):
        if rng is not None:
            return rng.choice(options)
        else:
            return random.choice(options)

    # 3. Obstacle placement logic
    available_red = list(red_fields)
    available_green = list(green_fields)
    
    # Obstacles are placed only in West, North, and East sections (not South)
    sections = ["Westen", "Norden", "Osten"]
    for section in sections:
        # Choose scenario: 0 (no obstacle), 1 (middle), 2 (one side), 3 (both sides)
        scenario = choose([0, 1, 2, 3])
        
        slots = []
        if scenario == 1:
            slots = [0.0]
        elif scenario == 2:
            slots = [choose([-0.5, 0.5])]
        elif scenario == 3:
            slots = [-0.5, 0.5]
            
        for s in slots:
            # Determine lateral displacement
            if section == "Westen":
                d = choose([-0.9, -1.1])
            else:
                d = choose([0.9, 1.1])
                
            # Calculate global coordinates
            if section == "Westen":
                x, y = d, s
            elif section == "Norden":
                x, y = s, d
            elif section == "Osten":
                x, y = d, s
                
            # Choose color based on available physical boxes
            chosen_color = None
            if available_red and available_green:
                chosen_color = choose(["red", "green"])
            elif available_red:
                chosen_color = "red"
            elif available_green:
                chosen_color = "green"
                
            if chosen_color == "red":
                field = available_red.pop()
                field.setSFVec3f([x, y, 0.05])
            elif chosen_color == "green":
                field = available_green.pop()
                field.setSFVec3f([x, y, 0.05])
