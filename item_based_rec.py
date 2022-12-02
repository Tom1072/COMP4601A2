import heapq
from math import sqrt
from copy import deepcopy


class ItemBasedRecommender:
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

        self.current_avg_ratings = deepcopy(self.avg_ratings)

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
            self.current_avg_ratings[user] = sum(
                filtered_ratings) / len(filtered_ratings) if len(filtered_ratings) > 0 else 0
        else:
            self.current_avg_ratings[user] = self.avg_ratings[user]

    def avg(self, user: int) -> float:
        """Update average rating a user"""
        return self.current_avg_ratings[user]

    def sim(self, a_idx: int, b_idx: int) -> float:
        """
        Calculate the similarity between two items, assume that avg_ratings is up-to-date
        @param a: item a index
        @param b: item b index
        @return similarity between a and b
        """
        a = self.current_transpose_matrix[a_idx]
        b = self.current_transpose_matrix[b_idx]

        sum_both = sum_a = sum_b = 0

        for u in range(len(a)):
            if a[u] == self.no_review or b[u] == self.no_review:
                continue

            avg_val = self.avg(u)
            sum_both += (a[u] - avg_val) * (b[u] - avg_val)
            sum_a += pow(a[u] - avg_val, 2)
            sum_b += pow(b[u] - avg_val, 2)

        result = None
        if (sum_a == 0 or sum_b == 0):
            result = 0
        else:
            result = sum_both / (sqrt(sum_a) * sqrt(sum_b))

        return result

    def pred_with_neighborhood_size(self, u: int, p: int, neighborhood_size: int) -> float:
        """Predict the rating of user u on item p with the neighborhood of highest similarity
        @param u: user index
        @param p: item index
        @param neighborhood_size: the size of neighborhood
        @return predicted rating
        """
        sim_heap = []
        for i in range(self.num_items):
            if i == p or self.current_matrix[u][i] == self.no_review:
                continue

            current_sim = self.sim(p, i)

            if len(sim_heap) < neighborhood_size:
                heapq.heappush(sim_heap, (current_sim, i))
            else:
                if current_sim > sim_heap[0][0]:
                    heapq.heappushpop(sim_heap, (current_sim, i))
        return self.pred_based_on_chosen_neighbors(sim_heap, u, p)

    def pred_with_sim_threshold(self, u: int, p: int, sim_threshold: int) -> float:
        """Predict the rating of user u on item p with neighbors that have similarity higher than sim_threshold
        @param u: user index
        @param p: item index
        @param sim_threshold: the threshold of similarity
        @return predicted rating
        """
        neighbors = []
        for i in range(self.num_items):
            if i == p or self.current_matrix[u][i] == self.no_review:
                continue

            current_sim = self.sim(p, i)
            if (current_sim > sim_threshold):
                neighbors.append((current_sim, i))

        return self.pred_based_on_chosen_neighbors(neighbors, u, p)
    
    def pred_with_absolute_sim_threshold(self, u: int, p: int, absolute_sim_threshold: int) -> float:
        """Predict the rating of user u on item p with neighbors that have absolute similarity higher than sim_threshold
        @param u: user index
        @param p: item index
        @param absolute_sim_threshold: the absolute threshold of similarity
        @return predicted rating
        """
        neighbors = []
        for i in range(self.num_items):
            if i == p or self.current_matrix[u][i] == self.no_review:
                continue

            current_sim = self.sim(p, i)
            if (abs(current_sim) > absolute_sim_threshold):
                neighbors.append((current_sim, i))

        return self.pred_based_on_chosen_neighbors(neighbors, u, p)

    def pred_based_on_chosen_neighbors(self, neighbors: list, u: int, p: int) -> float:
        """Predict the rating of user u on item p
        @param neighbors list(tuple(sim_val, index))): the list of users that are similar to user a and their similarity values
        @param u: user index
        @param p: item index
        @return predicted rating
        """
        numerator = 0
        denominator = 0
        for sim_val, i in neighbors:
            numerator += sim_val * self.current_matrix[u][i]
            denominator += sim_val

        if denominator == 0:
            return self.avg(u)
        else:
            return numerator / denominator
