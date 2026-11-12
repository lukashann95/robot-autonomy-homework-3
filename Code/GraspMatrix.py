import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


def grasp_column(n, p):
    torque = p[0] * n[1] - p[1] * n[0]
    return np.array([n[0], n[1], torque])

n_a = np.array([0, -1])
p_a = np.array([-0.05, 0.05])
g_a = grasp_column(n_a, p_a)

n_b = np.array([0, -1])
p_b = np.array([0.05, 0.05])
g_b = grasp_column(n_b, p_b)

n_c = np.array([1, 0])
p_c = np.array([-0.1, 0.0])
g_c = grasp_column(n_c, p_c)

n_d = np.array([-1, 0])
p_d = np.array([0.1, 0.0])
g_d = grasp_column(n_d, p_d)

n_e = np.array([0, 1])
p_e = np.array([-0.05, -0.05])
g_e = grasp_column(n_e, p_e)

n_f = np.array([0, 1])
p_f = np.array([0.05, -0.05])
g_f = grasp_column(n_f, p_f)

G = np.column_stack((g_a, g_b, g_c, g_d, g_e, g_f))

print(G)

plt.figure()
plt.plot(g_a[0], g_a[2], 'o', label='g_a')
plt.plot(g_b[0], g_b[2], 'o', label='g_b')
plt.plot(g_c[0], g_c[2], 'o', label='g_c')
plt.plot(g_d[0], g_d[2], 'o', label='g_d')
plt.plot(g_e[0], g_e[2], 'o', label='g_e')
plt.plot(g_f[0], g_f[2], 'o', label='g_f')

plt.legend()
plt.xlabel('F-X')
plt.ylabel(r'$T-\theta$')
plt.suptitle(r'Columns of Grasp Matrix in F-X vs $T-\theta$')
# plt.show()

plt.figure()
plt.plot(g_a[1], g_a[2], 'o', label='g_a')
plt.plot(g_b[1], g_b[2], 'o', label='g_b')
plt.plot(g_c[1], g_c[2], 'o', label='g_c')
plt.plot(g_d[1], g_d[2], 'o', label='g_d')
plt.plot(g_e[1], g_e[2], 'o', label='g_e')
plt.plot(g_f[1], g_f[2], 'o', label='g_f')

plt.legend()
plt.xlabel('F-Y')
plt.ylabel(r'$T-\theta$')
plt.suptitle(r'Columns of Grasp Matrix in F-Y vs $T-\theta$')
# plt.show()

def positively_spans(G):
    result = linprog(
        c=np.zeros(G.shape[1]),
        A_eq=G,
        b_eq=np.zeros(G.shape[0]),
        bounds=[(1, None)] * G.shape[1],
        method="highs",
    )

    positively_spans = (
        np.linalg.matrix_rank(G) == 3
        and result.success
    )

    print("Positively spans wrench space:", positively_spans)

print('Full G')
positively_spans(G)
print('Without point A')
positively_spans(G[:, 1:])
print('without point B')
positively_spans(G[:, [0, 2, 3, 4, 5]])
print('without point C')
positively_spans(G[:, [0, 1, 3, 4, 5]])
print('without point D')
positively_spans(G[:, [0, 1, 2, 4, 5]])
print('without point E')
positively_spans(G[:, [0, 1, 3, 4, 5]])
print('without point F')
positively_spans(G[:, [0, 1, 2, 3, 5]])
print('without point G')
positively_spans(G[:, :-1])

# ---------------------------------------------------

def grasp_metric(G):
    return np.sqrt(np.linalg.det(G @ G.T))

n_s = np.array([0, 1])
p_s = np.array([0, -0.1])
g_s = grasp_column(n_s, p_s)

n_a1_0 = np.array([1, 0])
n_a1_1 = np.array([0, -1])
p_a1 = np.array([0, 0.1])
g_a1_0 = grasp_column(n_a1_0, p_a1)
g_a1_1 = grasp_column(n_a1_1, p_a1)

n_a2_0 = np.array([-1, 0])
n_a2_1 = np.array([0, -1])
p_a2 = np.array([0, 0.1])
g_a2_0 = grasp_column(n_a2_0, p_a2)
g_a2_1 = grasp_column(n_a2_1, p_a2)

G_a = np.column_stack((g_s, g_a1_0, g_a1_1, g_a2_0, g_a2_1))
print(f'Grasp metric for a: {grasp_metric(G_a)}')


n_b1_0 = np.array([1, 0])
n_b1_1 = np.array([0, -1])
p_b1 = np.array([0.1, 0])
g_b1_0 = grasp_column(n_b1_0, p_b1)
g_b1_1 = grasp_column(n_b1_1, p_b1)

n_b2_0 = np.array([-1, 0])
n_b2_1 = np.array([0, -1])
p_b2 = np.array([-0.1, 0,])
g_b2_0 = grasp_column(n_b2_0, p_b2)
g_b2_1 = grasp_column(n_b2_1, p_b2)

G_b = np.column_stack((g_s, g_b1_0, g_b1_1, g_b2_0, g_b2_1))
print(f'Grasp metric for b: {grasp_metric(G_b)}')


n_c1_0 = np.array([1, 0])
n_c1_1 = np.array([0, -1])
p_c1 = np.array([-0.2, -0.1])
g_c1_0 = grasp_column(n_c1_0, p_c1)
g_c1_1 = grasp_column(n_c1_1, p_c1)

n_c2_0 = np.array([-1, 0])
n_c2_1 = np.array([0, -1])
p_c2 = np.array([0.2, -0.1,])
g_c2_0 = grasp_column(n_c2_0, p_c2)
g_c2_1 = grasp_column(n_c2_1, p_c2)

G_c = np.column_stack((g_s, g_c1_0, g_c1_1, g_c2_0, g_c2_1))
print(f'Grasp metric for c: {grasp_metric(G_c)}')
