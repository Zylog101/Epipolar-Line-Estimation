import cv2
import sys
import numpy as np

WIN_1 = "win_1"
WIN_2 = "win_2"
win_1_point_list = []
win_2_point_list = []
img1 = []
img2 = []
F = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def win2_line_draw_mouse_event_callback(event, x, y, flags, param):
    # print("win2_line_draw_mouse_event_callback")
    if event == cv2.EVENT_LBUTTONDBLCLK:
        point_right = (x, y, 1)
        cv2.circle(img2, (x, y), 2, (255, 0, 0), -1)
        # draw epipolar line
        draw_epipolar_line(point_right, False)
        compute_epipole()
        cv2.imshow(WIN_2, img2)


def draw_epipolar_line(point, right_image):
    global img1, img2, F
    # templist = []

    if right_image:
        P = np.matrix(point)
        l = F * P.transpose()
        draw_line(img2, l)
        # for x, y in zip(range(img2.shape[0]), range(img2.shape[1])):
        #     Pr_transpose = np.matrix([[x, y, 1]])
        #     temp = Pr_transpose * l
        #     if 0.1 > temp[0, 0] > -0.1:
        #         cv2.circle(img2, (x, y), 1, (255, 0, 0), -1)
        #         templist.append((x,y))
        cv2.imshow(WIN_2, img2)
    else:
        P = np.matrix(point)
        r = F.transpose() * P.transpose()
        draw_line(img1, r)
        # for x, y in zip(range(img2.shape[0]), range(img2.shape[1])):
        #     Pr_transpose = np.matrix([[x, y, 1]])
        #     temp = Pr_transpose * l
        #     if 0.1 > temp[0, 0] > -0.1:
        #         cv2.circle(img2, (x, y), 1, (255, 0, 0), -1)
        #         templist.append((x,y))
        cv2.imshow(WIN_1, img1)


def draw_line(img, l):
    # computing based on the values of coefficients
    # using Ax + By + C = 0
    x, y = img.shape
    x0, y0 = map(int, [0, -l[2] / l[1]])
    x1, y1 = map(int, [y, -(l[2] + l[0] * y) / l[1]])
    # x1,y1 = map(int, [y, -((l[0]*y)+l[2]) / l[1]])
    cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 1)


def win1_line_draw_mouse_event_callback(event, x, y, flags, param):
    # print("win1_line_draw_mouse_event_callback")
    if event == cv2.EVENT_LBUTTONDBLCLK:
        point_left = (x, y, 1)
        cv2.circle(img1, (x, y), 2, (0, 255, 0), -1)
        # draw epipolar line
        draw_epipolar_line(point_left, True)
        compute_epipole()
        cv2.imshow(WIN_1, img1)


def compute_epipole():
    F_transpose = F.transpose()
    v, d, u_transpose = np.linalg.svd(F_transpose)
    u = u_transpose.transpose()
    er = u[u.shape[0]-1, u.shape[1]-1]

    u, d, v_transpose = np.linalg.svd(F)
    v = v_transpose.transpose()
    el = v[v.shape[0]-1, v.shape[1]-1]
    print("left epipole {}", el)
    print("right epipole []", er)


def win1_mouse_event_callback(event, x, y, flags, param):
    # print("win1_mouse_event_callback")
    if event == cv2.EVENT_LBUTTONDBLCLK:
        point_right = (x, y, 1)
        cv2.circle(img1, (x, y), 2, (0, 255, 0), -1)

        win_1_point_list.append((x, y, 1))
        cv2.circle(img1, (x, y), 2, (255, 0, 0), -1)
        cv2.imshow(WIN_1, img1)

def win2_mouse_event_callback(event, x, y, flags, param):
    # print("win2_mouse_event_callback")
    if event == cv2.EVENT_LBUTTONDBLCLK:
        win_2_point_list.append((x, y, 1))
        cv2.circle(img2, (x, y), 2, (255, 0, 0), -1)
        cv2.imshow(WIN_2, img2)


def read_command_line():
    command_args_length = len(sys.argv)
    if command_args_length == 3:
        img1_name = sys.argv[1]
        img2_name = sys.argv[2]
        return img1_name, img2_name
    else:
        print("Failed To Read Command Line : usage Main.py image1name image2name")
        exit(-1)


def initialize_images(img1_name, img2_name, re_initialize):
    global img1, img2

    img1 = cv2.imread(img1_name)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow(WIN_1)
    img2 = cv2.imread(img2_name)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow(WIN_2)

    if re_initialize:
        cv2.setMouseCallback(WIN_1, win1_line_draw_mouse_event_callback)
        cv2.setMouseCallback(WIN_2, win2_line_draw_mouse_event_callback)
    else:
        cv2.setMouseCallback(WIN_1, win1_mouse_event_callback)
        cv2.setMouseCallback(WIN_2, win2_mouse_event_callback)
    return img1, img2


def display_images():
    cv2.imshow(WIN_1, img1)
    cv2.imshow(WIN_2, img2)
    return


def compute_mean_sd(point_list):
    mean_x = (np.mean(point_list, axis=0))[0]
    mean_y = (np.mean(point_list, axis=1))[1]
    mean = (mean_x, mean_y)
    sd_x = (np.std(point_list, axis=0))[0]
    sd_y = (np.std(point_list, axis=1))[1]
    sd = (sd_x, sd_y)
    return mean, sd


def normalize_point_lists(win_1_point_list, win_2_point_list):
    norm_point_list_1 = []
    norm_point_list_2 = []
    # normalize point list
    # compute mean and sd of corresponding point list
    mean, sd = compute_mean_sd(win_1_point_list)
    mean_prime, sd_prime = compute_mean_sd(win_2_point_list)
    M = np.matrix([[(1/sd[0]), 0, 0], [0, (1/sd[1]), 0], [0, 0, 1]]) * np.matrix([[1, 0, (-mean[0])], [0, 1, (-mean[1])], [0, 0, 1]])
    M_prime = np.matrix([[1/sd_prime[0], 0, 0], [0, 1/sd_prime[1], 0], [0, 0, 1]]) * np.matrix([[1, 0, -mean_prime[0]], [0, 1, -mean_prime[1]], [0, 0, 1]])

    for point in win_1_point_list:
        norm_point_list_1.append(M * np.matrix([[point[0]], [point[1]], [point[2]]]))
    for point in win_2_point_list:
        norm_point_list_2.append(M_prime * np.matrix([[point[0]], [point[1]], [point[2]]]))

    return norm_point_list_1, norm_point_list_2, M, M_prime


def compute_sys_of_eq_matrix(point_list_1, point_list_2):
    mat_line = []
    mat_list = []
    for point1, point2 in zip(point_list_1, point_list_2):
        p_1_1 = (point1[0].tolist()[0])[0]
        p_1_2 = (point1[1].tolist()[0])[0]

        p_2_1 = (point2[0].tolist()[0])[0]
        p_2_2 = (point2[1].tolist()[0])[0]
        mat_line = (p_1_1*p_2_1, p_1_1*p_2_2, p_1_1, p_1_2*p_2_1, p_1_2*p_2_2, p_1_2, p_2_1, p_2_2, 1)
        mat_list .append(mat_line)
    mat = np.matrix(mat_list)
    return mat


def normalized_points_fundamental_matrix(norm_point_list_1, norm_point_list_2):
    A = compute_sys_of_eq_matrix(norm_point_list_1,norm_point_list_2)
    u, d, vtranspose = np.linalg.svd(A, full_matrices=False)
    d[7] = 0
    # A_prime = np.dot(u, np.dot(np.diag(d), vtranspose))
    A_prime = u*np.diag(d) * vtranspose

    u, d, vtranspose = np.linalg.svd(A_prime)
    v = vtranspose.transpose()[:, 8]
    v = np.matrix([[v[0, 0], v[1, 0], v[2, 0]], [v[3, 0], v[4, 0], v[5, 0]], [v[6, 0], v[7, 0], v[8, 0]]])
    return v


def find_fundamental_matrix(norm_point_list_1, norm_point_list_2, M, M_prime):
    F_prime = normalized_points_fundamental_matrix(norm_point_list_1, norm_point_list_2)
    F = M_prime.transpose() * F_prime * M
    return F




def main():
    global img1, img2, F
    img1_name, img2_name = read_command_line()
    img1, img2 = initialize_images(img1_name, img2_name, False)
    display_images()
    cv2.waitKey()
    norm_point_list_1, norm_point_list_2, M, M_prime = normalize_point_lists(win_1_point_list, win_2_point_list)
    F = find_fundamental_matrix(norm_point_list_1, norm_point_list_2, M, M_prime)
    # re initialize image
    cv2.destroyAllWindows()
    img1, img2 = initialize_images(img1_name, img2_name, True)
    display_images()
    cv2.waitKey()


if __name__ == '__main__':
    main()
