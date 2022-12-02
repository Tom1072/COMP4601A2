import heapq
from math import sqrt
from itertools import combinations
from copy import deepcopy


class UserBasedRecommender:
    def __init__(self, filename: str, no_review: int) -> None:
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
        self.current_matrix = deepcopy(matrix)

        self.transpose_matrix = [[matrix[j][i] for j in range(
            len(matrix))] for i in range(len(matrix[0]))]
        self.current_transpose_matrix = deepcopy(self.transpose_matrix)

        # Initialize the average rating of each user
        self.avg_ratings = []
        for u in range(self.num_users):
            filtered_ratings = list(
                filter(lambda x: x != self.no_review, self.matrix[u]))
            self.avg_ratings.append(
                sum(filtered_ratings) / len(filtered_ratings) if len(filtered_ratings) > 0 else 0)

        self.avg_ratings_thread = deepcopy(self.avg_ratings)

    def update_ratings(self, user: int, item: int, rating: float) -> None:
        """ Update the rating, avg_ratings, and sim_matrix
        @param user: the user index
        @param item: the item index
        @param rating: the new rating
        """
        # Update the rating current_matrix
        self.current_matrix[user][item] = self.current_transpose_matrix[item][user] = rating

        if rating == self.no_review:
            # Update the average rating of the user
            filtered_ratings = list(
                filter(lambda x: x != self.no_review, self.current_matrix[user]))
            self.avg_ratings_thread[user] = sum(
                filtered_ratings) / len(filtered_ratings) if len(filtered_ratings) > 0 else 0
        else:
            self.avg_ratings_thread[user] = self.avg_ratings[user]

    def avg(self, user: int) -> float:
        """Return the average rating of the user"""
        return self.avg_ratings_thread[user]

    def sim(self, a_idx: int, b_idx: int) -> float:
        """Calculate the similarity between two users
        @param a_idx (int): user a index
        @param b_idx (int): user b index
        @return float: similarity between user a and user b
        """
        a = self.current_matrix[a_idx]
        b = self.current_matrix[b_idx]
        sum_both = sum_a = sum_b = 0

        for p in range(len(a)):
            if a[p] == self.no_review or b[p] == self.no_review:
                continue

            a_avg = self.avg(a_idx)
            b_avg = self.avg(b_idx)

            sum_both += (a[p] - a_avg) * (b[p] - b_avg)
            sum_a += pow(a[p] - a_avg, 2)
            sum_b += pow(b[p] - b_avg, 2)

        result = None
        if (sum_a == 0 or sum_b == 0):
            result = 0
        else:
            result = sum_both / (sqrt(sum_a) * sqrt(sum_b))

        return result

    def pred_with_neighborhood_size(self, a: int, p: int, neighborhood_size: int) -> float:
        """Predict the rating of user a on item p with the neighborhood of highest similarity
        @param u: user index
        @param p: item index
        @param neighborhood_size: the size of the neighborhood of highest similarity
        @param sim_threshold: the threshold of similarity
        @return predicted rating
        """
        sim_heap = []
        for b in range(self.num_users):
            if b == a or self.current_matrix[b][p] == self.no_review:
                continue
            
            current_sim = self.sim(a, b)

            if len(sim_heap) < neighborhood_size:
                heapq.heappush(sim_heap, (current_sim, b))
            else:
                if len(sim_heap) > 0:
                    if current_sim > sim_heap[0][0]:
                        heapq.heappushpop(sim_heap, (current_sim, b))
        return self.pred_based_on_chosen_neighbors(sim_heap, a, p)

    def pred_with_sim_threshold(self, a: int, p: int, sim_threshold: int) -> float:
        """Predict the rating of user a on item p with neighbors that have similarity higher than sim_threshold
        @param u: user index
        @param p: item index
        @param neighborhood_size: the size of the neighborhood of highest similarity
        @param sim_threshold: the threshold of similarity
        @return predicted rating
        """
        neighbors = []
        for b in range(self.num_users):
            if b == a or self.current_matrix[b][p] == self.no_review:
                continue
            
            current_sim = self.sim(a, b)
            if (current_sim >= sim_threshold):
                neighbors.append((current_sim, b))

        return self.pred_based_on_chosen_neighbors(neighbors, a, p)

    def pred_with_absolute_sim_threshold(self, a: int, p: int, absolute_sim_threshold: int) -> float:
        """Predict the rating of user a on item p with neighbors that have absolute similarity higher than sim_threshold
        @param u: user index
        @param p: item index
        @param neighborhood_size: the size of the neighborhood of highest similarity
        @param sim_threshold: the threshold of similarity
        @return predicted rating
        """
        neighbors = []
        for b in range(self.num_users):
            if b == a or self.current_matrix[b][p] == self.no_review:
                continue
            
            current_sim = self.sim(a, b)
            if (abs(current_sim) >= absolute_sim_threshold):
                neighbors.append((current_sim, b))

        return self.pred_based_on_chosen_neighbors(neighbors, a, p)

    def pred_based_on_chosen_neighbors(self, neighbors: list, a: int, p: int) -> float:
        """Predict the rating of user u on item p
        @param neighbors list(tuple(sim_val, index))): the list of users that are similar to user a and their similarity values
        @param a: user index
        @param p: item index
        @return predicted rating
        """
        numerator = 0
        denominator = 0
        for sim_val, b in neighbors:
            numerator += sim_val * \
                (self.current_matrix[b][p] - self.avg(b))
            denominator += sim_val

        if denominator == 0:
            return self.avg(a)
        else:
            return self.avg(a) + (numerator / denominator)