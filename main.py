from imported_libs import *

pygame.init()
pygame.display.set_caption('3D Engine')
#screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
#w, h = screen.get_size()
screen = pygame.display.set_mode((500, 500))
w, h = 500, 500
cx, cy = int(w / 2), int(h / 2)
clock = pygame.time.Clock()
fov = 90

verts = []
faces = []

filepath = 'objects/pokus3.obj'
with open(filepath) as fp:
    line = fp.readline()
    cnt = 1
    while line:
        a = line.strip().split(" ")

        if a[0] == "v":
            v = [float(a[1]), float(a[2]), float(a[3])]
            verts.append(v)

        elif a[0] == "f":
            f = [float(a[1]), float(a[2]), float(a[3])]
            faces.append(f)

        line = fp.readline()
        cnt += 1

class Shape():

    def __init__(self):
        t = np.zeros((len(faces), 3, 4))
        for i in range(len(faces)):
            v1 = verts[int(faces[i][0] - 1)] + [1]
            v2 = verts[int(faces[i][1] - 1)] + [1]
            v3 = verts[int(faces[i][2] - 1)] + [1]
            t[i][0] = v1
            t[i][1] = v2
            t[i][2] = v3
        self.triangles = t


class Camera():

    def __init__(self):
        self.pos = np.array([0, 0, 0])
        self.dir = np.array([0, 0, 1])
        self.yaw = 0

    def update(self, key, dt):

        forward = np.dot(camera.dir, 8*dt)
        if key[pygame.K_w]:
            self.pos = np.add(self.pos, forward)

        if key[pygame.K_s]:
            self.pos = np.subtract(self.pos, forward)

        if key[pygame.K_a]:
            self.yaw -= 2*dt

        if key[pygame.K_d]:
            self.yaw += 2*dt


camera = Camera()
shape = Shape()

z1 = 0.1
z2 = 1000
r = h / w
f = 1 / tan(radians(fov / 2))

projection = np.array([[r * f, 0, 0, 0],
                       [0, f, 0, 0],
                       [0, 0, z2 / (z2 - z1), 1],
                       [0, 0, (-z1 * z2) / (z2 - z1), 0]])

theta = 0
running = True
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT: pygame.quit(); sys.exit(); running = False
        if event.type == pygame.KEYDOWN:
            global_key = pygame.key.get_pressed()
            if global_key[pygame.K_ESCAPE]:
                running = False
                pygame.quit()
                sys.exit()

    screen.fill((0, 0, 0))

    rotation_Z = rotate_Z(theta)
    rotation_X = rotate_X(0)
    rotation_Y = rotate_Y(3/2*pi + theta)

    theta += 0.02

    dt = clock.tick() / 1000
    key = pygame.key.get_pressed()
    camera.update(key, dt)

    Up = np.array([0, 1, 0])
    FW = np.array([0,0,1])
    camera_rotation = rotate_Y_3(camera.yaw)
    camera.dir = np.dot(camera_rotation, FW)
    FW = np.add(camera.pos, camera.dir)

    camera_matrix = matrix_pointat(camera.pos, FW, Up)
    #view_matrix = matrix_inverse(camera_matrix)
    view_matrix = np.linalg.inv(camera_matrix)

    depth_list = []

    for triangle in shape.triangles:

        triangle_translated = np.zeros((3,4))

        for i in range(3):
            triangle_translated[i] = np.dot(np.dot(triangle[i], rotation_Y), rotation_X)
            triangle_translated[i, 2] = np.add(triangle_translated[i, 2], 6)

        vector_1 = create_vector(triangle_translated[1], triangle_translated[0])
        vector_2 = create_vector(triangle_translated[2], triangle_translated[0])

        normal = np.cross(vector_1, vector_2)
        normal = normalize(normal)

        camera_ray = create_vector(triangle_translated[0], camera.pos)

        if np.dot(normal, camera_ray) < 0:
            light = [0, 0, -1]
            light = normalize(light)
            dot_product = np.dot(light, normal)
            a = mapX(max(dot_product, 0.1), 0, 1, 0, 255)

            triangle_viewed = np.zeros((3, 4))
            triangle_viewed[0] = np.dot(view_matrix, triangle_translated[0])
            triangle_viewed[1] = np.dot(view_matrix, triangle_translated[1])
            triangle_viewed[2] = np.dot(view_matrix, triangle_translated[2])

            triangle_projected = np.zeros((3, 3))
            for i in range(3):
                triangle_p = multiply_normalize(triangle_viewed[i], projection)
                triangle_projected[i] = triangle_p

            for i in range(3):
                triangle_projected[i, 0] *= -1
                triangle_projected[i, 1] *= -1

            x, y = [], []
            for i in range(3):
                triangle_projected[i, 0] += 1#np.add(triangle_projected[i, 0], 1)
                triangle_projected[i, 1] += 1#np.add(triangle_projected[i, 1], 1)
                triangle_projected[i, 0] *= 0.5 * w
                triangle_projected[i, 1] *= 0.5 * h
                x.append(triangle_projected[i, 0])
                y.append(triangle_projected[i, 1])

            z = (triangle_translated[0][2] + triangle_translated[1][2] + triangle_translated[2][2]) / 3
            depth_list.append([z, [x[0], y[0]], [x[1], y[1]], [x[2], y[2]], a])

        depth_list = sorted(depth_list, reverse=1)

        for i in range(len(depth_list)):
            pygame.draw.polygon(screen, (depth_list[i][4], depth_list[i][4], depth_list[i][4]),
                                ((depth_list[i][1][0], depth_list[i][1][1]),
                                 (depth_list[i][2][0], depth_list[i][2][1]),
                                 (depth_list[i][3][0], depth_list[i][3][1])), 0)

    pygame.display.flip()
