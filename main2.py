from imported_libs import *

pygame.init()
pygame.display.set_caption('3D Engine')
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
w, h = screen.get_size()
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

def mapX(x, a, b, c, d):

    val = (x-a)/((b-a)/(d-c))+c

    return val

class Shape():

    def __init__(self):
        t = np.zeros((len(faces),3,4))
        for i in range(len(faces)):
            v1 = verts[int(faces[i][0] - 1)]+[1]
            v2 = verts[int(faces[i][1] - 1)]+[1]
            v3 = verts[int(faces[i][2] - 1)]+[1]
            t[i][0] = v1
            t[i][1] = v2
            t[i][2] = v3
        self.triangles = t

class Camera():

    def __init__(self):
        self.pos = np.array([0, 0, 0])

camera = Camera()

def multiply_normalize(a, b):
    a = np.dot(a,b)

    if a[3] != 0:
        a[0] /= a[3]
        a[1] /= a[3]
        a[2] /= a[3]

    return np.array([a[0], a[1], a[2]])

def create_vector(a, b):

    vector = np.array([a[0]-b[0],a[1]-b[1],a[2]-b[2]])
    return vector

def normalize(a):
    l = sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    a = np.array([a[0]/l, a[1]/l, a[2]/l])
    return a

shape = Shape()

z1 = 0.1
z2 = 1000
r = h/w
f = 1 / tan(radians(fov/2))

projection = np.array([[r*f, 0,              0, 0],
                       [0  , f,              0, 0],
                       [0  , 0,     z2/(z2-z1), 1],
                       [0  , 0, -z1*z2/(z2-z1), 0]])

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

    screen.fill((0,0,0))

    rotation_Z = np.array([[ cos(theta/2), sin(theta/2), 0, 0],
                           [-sin(theta/2), cos(theta/2), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    #rotation_X = np.array([[1, 0, 0, 0],
    #                       [0, cos(theta/2), sin(theta/2), 0],
    #                       [0,-sin(theta/2), cos(theta/2), 0],
    #                       [0, 0, 0, 1]])

    rotation_X = np.array([[1, 0, 0, 0],
                           [0, cos(2*pi/2), sin(2*pi/2), 0],
                           [0,-sin(2*pi/2), cos(2*pi/2), 0],
                           [0, 0, 0, 1]])

    rotation_Y = np.array([[cos(theta / 2), 0, sin(theta / 2), 0],
                           [0, 1, 0, 0],
                           [-sin(theta / 2), 0, cos(theta / 2), 0],
                           [0, 0, 0, 1]])

    theta += 0.01

    depth_list = []

    for triangle in shape.triangles:

        triangle_translated = np.zeros((3,4))

        for i in range(3):
            triangle_translated[i] = np.dot(np.dot(triangle[i], rotation_Y), rotation_X)
            triangle_translated[i, 2] = np.add(triangle_translated[i, 2], 10)

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

            triangle_p1 = multiply_normalize(triangle_translated[0], projection)
            triangle_p2 = multiply_normalize(triangle_translated[1], projection)
            triangle_p3 = multiply_normalize(triangle_translated[2], projection)

            triangle_projected = np.array([[triangle_p1[0], triangle_p1[1], triangle_p1[2]],
                                           [triangle_p2[0], triangle_p2[1], triangle_p2[2]],
                                           [triangle_p3[0], triangle_p3[1], triangle_p3[2]]])

            triangle_projected[0, 0] += 1; triangle_projected[0, 1] += 1
            triangle_projected[1, 0] += 1; triangle_projected[1, 1] += 1
            triangle_projected[2, 0] += 1; triangle_projected[2, 1] += 1

            triangle_projected[0, 0] *= 0.5 * w; triangle_projected[0, 1] *= 0.5 * h
            triangle_projected[1, 0] *= 0.5 * w; triangle_projected[1, 1] *= 0.5 * h
            triangle_projected[2, 0] *= 0.5 * w; triangle_projected[2, 1] *= 0.5 * h

            x1 = triangle_projected[0, 0]; y1 = triangle_projected[0, 1]
            x2 = triangle_projected[1, 0]; y2 = triangle_projected[1, 1]
            x3 = triangle_projected[2, 0]; y3 = triangle_projected[2, 1]
            z = (triangle_translated[0][2] + triangle_translated[1][2] + triangle_translated[2][2])/3

            depth_list.append([z,[x1,y1],[x2,y2],[x3,y3],a])

        depth_list = sorted(depth_list, reverse=1)

        for i in range(len(depth_list)):

            pygame.draw.polygon(screen, (depth_list[i][4], depth_list[i][4], depth_list[i][4]),
                                ((depth_list[i][1][0], depth_list[i][1][1]),
                                 (depth_list[i][2][0], depth_list[i][2][1]),
                                 (depth_list[i][3][0], depth_list[i][3][1])), 0)

    pygame.display.flip()