from math import sqrt
from copy import deepcopy
from progress.bar import IncrementalBar
import heapq

NO_REVIEW_FLAG = 0

def parse_input(filename):
    r, c = 0, 0
    users = None
    items = []
    matrix = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip().split(" ")
            if i == 0:
                r = int(line[0])
                c = int(line[1])
            elif i == 1:
                users = line
            elif i == 2:
                items = line
            elif i > 2:
                row = []
                for rating in line:
                    row.append(float(rating))
                matrix.append(row)

    return r, c, users, items, matrix


def average_rating(matrix: list):
    avg_ratings = []

    for i in range(len(matrix)):
        filtered_user = list(filter(lambda x: x != NO_REVIEW_FLAG, matrix[i]))
        sum_filtered = sum(filtered_user)
        avg_ratings.append([sum_filtered / len(filtered_user), sum_filtered, len(filtered_user)])

    return avg_ratings


def sim(a: list, b: list, avg_ratings: list):
    sum_both = sum_a = sum_b = 0
    for u in range(len(a)):
        if a[u] == NO_REVIEW_FLAG or b[u] == NO_REVIEW_FLAG:
            continue

        sum_both += (a[u] - avg_ratings[u][0]) * (b[u] - avg_ratings[u][0])
        sum_a += pow(a[u] - avg_ratings[u][0], 2)
        sum_b += pow(b[u] - avg_ratings[u][0], 2)

    return 0 if (sum_a == 0 or sum_b == 0) else sum_both / (sqrt(sum_a) * sqrt(sum_b))


def pred(u: int, p: int, matrix: list, item_num: int, neighborhood_size: int, zipped_matrix, average_ratings):
    numerator = 0
    denominator = 0
    sim_heap = []
    # sim_heap2 = []

    for i in range(item_num):
        if i == p or matrix[u][i] == NO_REVIEW_FLAG:
            continue
        current_sim = sim(zipped_matrix[i], zipped_matrix[p], average_ratings)
        if current_sim < 0:
            continue
        sim_heap.append((current_sim, i))
        # if len(sim_heap) < neighborhood_size:
        #     heapq.heappush(sim_heap, (current_sim, i))
        # else:
        #     if current_sim > sim_heap[0][0]:
        #         heapq.heappushpop(sim_heap, (current_sim, i))
    
    sim_heap2 = sorted(sim_heap, reverse=True)[:neighborhood_size]
    # sim_heap = sorted(sim_heap, reverse=True)[:neighborhood_size]
    # sim_heap = sorted(sim_heap, key=lambda x: x[0], reverse=True)[:neighborhood_size]
    # print("\n")
    # print(sim_heap)
    # print(sim_heap2)

    for s, i in sim_heap:
        numerator += s * matrix[u][i]
        denominator += s

    result = average_ratings[u][0] if denominator == 0 else numerator / denominator
    if result < 1: return 1
    if result > 5: return 5
    return result


def compute_mae(matrix, users, items, rec_matrix):
    """ Compute the mean absolute error of the prediction
    @return: the mean absolute error
    """
    numerator = 0
    denominator = 0
    for user in range(len(users)):
        for item in range(len(items)):
            if (matrix[user][item] == NO_REVIEW_FLAG):
                continue
            numerator += abs(rec_matrix[user][item] - matrix[user][item])
            denominator += 1

    # print(denominator)
    return numerator / denominator


def recommend(users, items, matrix, neighborhood_size):
    avg_ratings = average_rating(matrix)
    zipped_matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    pred_matrix = deepcopy(matrix)

    with IncrementalBar('Calculation predictions', max=len(users)*len(items), suffix='[%(index)d/%(max)d | %(elapsed_td)s/%(eta_td)s]') as bar:
        for i in range(len(users)):
            for a in range(len(items)):
                bar.next()
                if matrix[i][a] == NO_REVIEW_FLAG: continue

                avg_ratings[i][1] -= matrix[i][a]
                avg_ratings[i][2] -= 1
                avg_ratings[i][0] = avg_ratings[i][1] / avg_ratings[i][2]
                prev = matrix[i][a]
                matrix[i][a] = zipped_matrix[a][i] = 0

                pred_matrix[i][a] = pred(i, a, matrix, len(items), neighborhood_size, zipped_matrix, avg_ratings)

                matrix[i][a] = zipped_matrix[a][i] = prev
                avg_ratings[i][1] += matrix[i][a]
                avg_ratings[i][2] += 1
                avg_ratings[i][0] = avg_ratings[i][1] / avg_ratings[i][2]

        bar.finish()

    return compute_mae(matrix, users, items, pred_matrix)


def test_prev_lab():
    _, _, users, items, matrix = parse_input("lab7_files/test.txt")
    avg_ratings = average_rating(matrix)
    zipped_matrix = list(zip(*matrix))
    calculated_map = {}

    print("\nPrediction Matrix: ")
    pred_matrix = deepcopy(matrix)
    for i in range(len(users)):
        print(f"{users[i]}: [", end="")
        for j in range(len(items)):
            if matrix[i][j] == NO_REVIEW_FLAG:
                prediction = pred(i, j, matrix, len(items), 2,
                                  zipped_matrix, avg_ratings, calculated_map)
                print(f"{prediction}", end=", ")
                pred_matrix[i][j] = prediction
            else:
                print(f"{matrix[i][j]}", end=", ")
        print("]")

if __name__ == "__main__":
  row, col, users, items, matrix = parse_input("parsed-data-trimmed.txt")
  neighborhood_size = 5
  print(f"Matrix {row} x {col}\nNeighborhood size {neighborhood_size}")
  MAE = recommend(users, items, matrix, neighborhood_size)
  print("MAE:", MAE)