import heapq
from math import sqrt
from itertools import combinations
from copy import deepcopy


class UserBasedRecommender:
    def __init__(self, filename: str, num_threads: int, no_review: int) -> None:
        self.num_threads = num_threads
        self.no_review = no_review
        self.parse_input(filename)

    def parse_input(self, filename: str) -> None:
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

        self.row = self.num_users = r
        self.col = self.num_items = c
        self.users = users
        self.items = items

        self.matrix = matrix
        self.matrices = [deepcopy(matrix) for _ in range(self.num_threads)]

        self.transpose_matrix = [[matrix[j][i] for j in range(
            len(matrix))] for i in range(len(matrix[0]))]
        self.transpose_matrices = [
            deepcopy(self.transpose_matrix) for _ in range(self.num_threads)]

        # Initialize the average rating of each user
        self.avg_ratings = []
        for u in range(self.num_users):
            filtered_ratings = list(
                filter(lambda x: x != self.no_review, self.matrix[u]))
            self.avg_ratings.append(
                sum(filtered_ratings) / len(filtered_ratings) if len(filtered_ratings) > 0 else 0)

        self.avg_ratings_thread = [
            deepcopy(self.avg_ratings) for _ in range(self.num_threads)]

    def update_ratings(self, thread_id: int, user: int, item: int, rating: float) -> None:
        """ Update the rating, avg_ratings, and sim_matrix
        @param user: the user index
        @param item: the item index
        @param rating: the new rating
        """
        # Update the rating matrices
        self.matrices[thread_id][user][item] = self.transpose_matrices[thread_id][item][user] = rating

        if rating == self.no_review:
            # Update the average rating of the user
            filtered_ratings = list(
                filter(lambda x: x != self.no_review, self.matrices[thread_id][user]))
            self.avg_ratings_thread[thread_id][user] = sum(
                filtered_ratings) / len(filtered_ratings) if len(filtered_ratings) > 0 else 0
        else:
            self.avg_ratings_thread[thread_id][user] = self.avg_ratings[user]

    def avg(self, thread_id, user: int) -> float:
        """Return the average rating of the user"""
        return self.avg_ratings_thread[thread_id][user]

    def sim(self, thread_id: int, a_idx: int, b_idx: int) -> float:
        """Calculate the similarity between two users
        @param thread_id (int): _description_
        @param a_idx (int): user a index
        @param b_idx (int): user b index
        @return float: similarity between user a and user b
        """
        a = self.matrices[thread_id][a_idx]
        b = self.matrices[thread_id][b_idx]
        sum_both = sum_a = sum_b = 0

        for p in range(len(a)):
            if a[p] == self.no_review or b[p] == self.no_review:
                continue

            a_avg = self.avg(thread_id, a_idx)
            b_avg = self.avg(thread_id, b_idx)

            sum_both += (a[p] - a_avg) * (b[p] - b_avg)
            sum_a += pow(a[p] - a_avg, 2)
            sum_b += pow(b[p] - b_avg, 2)

        result = None
        if (sum_a == 0 or sum_b == 0):
            result = 0
        else:
            result = sum_both / (sqrt(sum_a) * sqrt(sum_b))

        return result

    def pred(self, thread_id: int, a: int, p: int, neighborhood_size: int, sim_threshold: float) -> float:
        """Predict the rating of user u on item p
        @param u: user index
        @param p: item index
        @param neighborhood_size: the size of the neighborhood of highest similarity
        @param sim_threshold: the threshold of similarity
        @return predicted rating
        """
        numerator = 0
        denominator = 0

        sim_heap = []
        for b in range(self.num_users):
            if b == a or self.matrices[thread_id][b][p] == self.no_review:
                continue

            current_sim = self.sim(thread_id, a, b)
            if current_sim < sim_threshold:
                continue

            if len(sim_heap) < neighborhood_size:
                heapq.heappush(sim_heap, (current_sim, b))
            else:
                if current_sim > sim_heap[0][0]:
                    heapq.heappushpop(sim_heap, (current_sim, b))

        for sim_val, b in sim_heap:
            numerator += sim_val * \
                (self.matrices[thread_id][b][p] - self.avg(thread_id, b))
            denominator += sim_val

        if denominator == 0:
            return self.avg(thread_id, a)
        else:
            return self.avg(thread_id, a) + (numerator / denominator)
