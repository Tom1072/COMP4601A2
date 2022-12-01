from math import sqrt
from itertools import combinations
from pprint import pprint
from copy import deepcopy

def parse_input(filename):
  r, c= 0, 0
  users = None
  items = []
  matrix = []
  with open(filename, "r") as f:
    for i, line in enumerate(f):
      line = line.strip()
      # print(f"line {i}: {line}")
      if i == 0:
        r = int(line.split(" ")[0])
        c = int(line.split(" ")[1])
      elif i == 1:
        users = line.split(" ")
      elif i == 2:
        items = line.split(" ")
      elif i > 2:
        row = []
        for rating in line.split(" "):
          row.append(int(rating))
        matrix.append(row)

  return r, c, users, items, matrix


def average_rating(matrix: list):
  avg_ratings = []

  for i in range(len(matrix)):
    filtered_user = list(filter(lambda x: x != -1, matrix[i]))
    avg_ratings.append(sum(filtered_user) / len(filtered_user))

  return avg_ratings


def sim(a: int, b: int, matrix: list, avg_ratings: list):
  sum_both = sum_a = sum_b = 0
  for u in range(len(matrix)):
    if matrix[u][a] == -1 or matrix[u][b] == -1:
      continue
    
    sum_both += (matrix[u][a] - avg_ratings[u]) * (matrix[u][b] - avg_ratings[u])
    sum_a += pow(matrix[u][a] - avg_ratings[u], 2)
    sum_b += pow(matrix[u][b] - avg_ratings[u], 2)

  return sum_both / (sqrt(sum_a) * sqrt(sum_b))


def pred(u: int, p: int, matrix: list, sim_matrix: list, neighborhood_size: int):
  numerator = 0
  denominator = 0

  bs = []
  mapped_sim_matrix = [(x, i) for i, x in enumerate(sim_matrix[p])]
  mapped_sim_matrix = set(filter(lambda e: e[1] != p and matrix[u][e[1]] != -1 and e[0] >= 0, mapped_sim_matrix))
  neighborhood_size = min(neighborhood_size, len(mapped_sim_matrix))

  for i in range(neighborhood_size):
    max_val = max(mapped_sim_matrix, key=lambda e: e[0])
    mapped_sim_matrix.remove(max_val)
    bs.append(max_val[1]) # Only take the index i

  for i in bs:
    numerator += sim_matrix[i][p] * matrix[u][i]
    denominator += sim_matrix[i][p]

  return numerator / denominator

if __name__ == "__main__":
  row, col, users, items, matrix = parse_input("testa.txt")
  avg_ratings = average_rating(matrix)
  sim_matrix = [[0 for _ in range(col)] for __ in range(col)]

  for a, b in combinations(range(len(items)), 2):
    sim_matrix[b][a] = sim_matrix[a][b] = sim(a, b, matrix, avg_ratings) 
    
  print("Similarity Matrix: ")
  for i, r in enumerate(sim_matrix):
    # print(f"{items[i]}: ", end="")
    print(f"{items[i]}: ", end="")
    print("[", end="")
    for j, e in enumerate(r):
      # print(f"{items[j]}: {e:5.2f}", end=", ")
      print(f"{e:5.2f}", end=", ")
    print("]")
  
  print("\nPrediction Matrix: ")
  pred_matrix = deepcopy(matrix)
  for i in range(len(users)):
    print(f"{users[i]}: [", end="")
    for j in range(len(items)):
      if matrix[i][j] == -1:
        prediction =  pred(i, j, matrix, sim_matrix, 2)
        print(f"{prediction:5.2f}", end=", ")
        pred_matrix[i][j] = prediction
      else:
        print(f"{matrix[i][j]:5.2f}", end=", ")
    print("]")