from math import sqrt
from itertools import combinations
from pprint import pprint
from copy import deepcopy

class ItemBasedRecommender:
  def __init__(self, filename: str, num_threads: int, no_review: int) -> None:
      self.num_threads = num_threads
      self.no_review = no_review
      self.parse_input(filename)

  def parse_input(self, filename):
    r, c= 0, 0
    users = None
    items = []
    matrix = []
    with open(filename, "r") as f:
      for i, line in enumerate(f):
        line = line.strip()
        print(f"line {i}: {line}")
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
    for p in range(len(matrix[0])):
      if matrix[a][p] == -1 or matrix[b][p] == -1:
        continue

      sum_both += (matrix[a][p] - avg_ratings[a]) * (matrix[b][p] - avg_ratings[b])
      sum_a += pow(matrix[a][p] - avg_ratings[a], 2)
      sum_b += pow(matrix[b][p] - avg_ratings[b], 2)

    return sum_both / (sqrt(sum_a) * sqrt(sum_b))


  def pred(a: int, p: int, matrix: list, sim_matrix: list, avg_ratings: list, neighborhood_size: int):
    numerator = 0
    denominator = 0
    bs = sorted([(x, i) for i, x in enumerate(sim_matrix[a])], reverse=True)
    bs = list(filter(lambda e: e != a, map(lambda e: e[1], bs)))
    bs = bs[:neighborhood_size]

    for b in bs:
      numerator += sim_matrix[a][b] * (matrix[b][p] - avg_ratings[b])
      denominator += sim_matrix[a][b]

    return avg_ratings[a] + numerator / denominator
    

  if __name__ == "__main__":
    row, col, users, items, matrix = parse_input("test3.txt")
    avg_ratings = average_rating(matrix)
    sim_matrix = [[0 for _ in range(row)] for __ in range(row)]

    for a, b in combinations(range(len(users)), 2):
      sim_matrix[b][a] = sim_matrix[a][b] = sim(a, b, matrix, avg_ratings) 
      

    # pprint(sim_matrix)
    print("Similarity Matrix: ")
    for i, r in enumerate(sim_matrix):
      print(f"{users[i]}: ", end="")
      print("[", end="")
      for j, e in enumerate(r):
        print(f"{users[j]}: {e:5.2f}", end=", ")
      print("]")
    
    print("\nPrediction Matrix: ")
    pred_matrix = deepcopy(matrix)
    for i in range(len(users)):
      print(f"[{users[i]}: ", end="")
      for j in range(len(items)):
        if matrix[i][j] == -1:
          prediction =  pred(i, j, matrix, sim_matrix, avg_ratings, 2)
          print(f"{items[j]}: {prediction:5.2f}", end=", ")
          pred_matrix[i][j] = prediction
        else:
          print(f"{items[j]}: {matrix[i][j]:5.2f}", end=", ")
      print("]")