import random
import json

N = 10000

scene = {
    "integrator": {
        "TYPE": "path",
        "DEPTH": 8,
        "ITERATIONS": 8124
    },
    "camera": {
        "TYPE": "pinhole",
        "FOVY": 45.0,
        "EYE": [0.0, 0.0, -5.0],
        "LOOKAT": [0.0, 0.0, 0.0],
        "UP": [0.0, 1.0, 0.0],
        "film": {
            "WIDTH": 1920,
            "HEIGHT": 1080,
            "COLOR_SPACE": "rgb"
        },
        "sampler": {
            "TYPE": "independent"
        },
        "reconstruction_filter": {
            "TYPE": "box"
        }
    },
    "materials": {
        "light": {
            "TYPE": "emitting",
            "RADIANCE": {"RGB": [1.0, 1.0, 1.0], "EMITTANCE": 5.0}
        },
        "mirror": {
            "TYPE": "mirror",
            "ALBEDO": {"RGB": [0.98, 0.98, 0.98] }
        },
        "diffuse_red": {
            "TYPE": "diffuse",
            "ALBEDO": {"RGB": [0.86, 0.35, 0.35] }
        }
    },
    "objects": []
}

base_arm = {
    "TYPE": "instance",
    "NAME": "armadillo",
    "INSTANCE": {
        "TYPE": "gltf",
        "MATERIAL": "diffuse_red",
        "PATH": "armadillo.gltf",
        "FACE_NORMALS": False
    }
}

base_dragon = {
    "TYPE": "instance",
    "NAME": "dragon",
    "INSTANCE": {
        "TYPE": "gltf",
        "MATERIAL": "mirror",
        "PATH": "dragon.gltf",
        "FACE_NORMALS": False
    }
}
scene["objects"].append(base_arm)
scene["objects"].append(base_dragon)

for _ in range(N):
    scale = random.uniform(0.05, 0.3)
    scale2 = random.uniform(0.15, 0.5)

    instance1 = {
        "TYPE": "instance",
        "SOURCE": "dragon",
        "TRANS": [random.uniform(-10, 10) for _ in range(3)],
        "ROTAT": [random.uniform(0, 360) for _ in range(3)],
        "SCALE": [scale2, scale2, scale2]
    }
    instance2 = {
        "TYPE": "instance",
        "SOURCE": "armadillo",
        "TRANS": [random.uniform(-10, 10) for _ in range(3)],
        "ROTAT": [random.uniform(0, 360) for _ in range(3)],
        "SCALE": [scale, scale, scale]
    }
    scene["objects"].append(instance1)
    scene["objects"].append(instance2)

print(json.dumps(scene, indent=2))