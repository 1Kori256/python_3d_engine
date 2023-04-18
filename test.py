vector_1 = create_vector(triangle_translated[1], triangle_translated[0])
vector_2 = create_vector(triangle_translated[2], triangle_translated[0])

normal = [0, 0, 0]

normal[0] = float(vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1])
normal[1] = float(vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2])
normal[2] = float(vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0])
length = (sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2))

normal[0] /= length
normal[1] /= length
normal[2] /= length