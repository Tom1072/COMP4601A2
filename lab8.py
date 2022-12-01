from item_based_rec import ItemBasedRecommender
from pprint import pprint
from copy import deepcopy
from itertools import combinations
from math import comb
from threading import Thread, Lock
from tqdm import tqdm


class RecommenderCrossValidator(ItemBasedRecommender):
    def __init__(self, filename, num_threads, no_review=0) -> None:
        super().__init__(filename, num_threads, no_review)
        self.rec_matrix = [[0 for entry in range(
            self.num_items)] for row in range(self.num_users)]
        self.rec_matrix_lock = Lock()

    def update_rec_matrix(self, user: int, item: int, pred_rating: float) -> None:
        if pred_rating > 5:
            pred_rating = 5
        elif pred_rating < 1:
            pred_rating = 1

        self.rec_matrix_lock.acquire()
        self.rec_matrix[user][item] = pred_rating
        self.rec_matrix_lock.release()

    def update_pred(self, thread_id: int, start_row: int, end_row: int, col: int, neighborhood_size: int, cross_val: bool = True) -> None:
        progress_bar = tqdm(total=(end_row - start_row) * self.num_items)
        progress_bar.set_description(f"Pred Thread {thread_id:2.0f}")
        # for user in tqdm(range(start_row, end_row)):
        for user in range(start_row, end_row):
            for item in range(col):
                progress_bar.update(1)
                if cross_val:
                    if self.matrix[user][item] == no_review:
                        continue
                    self.update_ratings(thread_id, user, item, self.no_review)
                    pred_rating = self.pred(
                        thread_id, user, item, neighborhood_size)
                    self.update_rec_matrix(user, item, pred_rating)
                    self.update_ratings(
                        thread_id, user, item, self.matrix[user][item])
                else:
                    if self.matrix[user][item] != no_review:
                        self.update_rec_matrix(
                            user, item, self.matrix[user][item])
                        continue
                    pred_rating = self.pred(
                        thread_id, user, item, neighborhood_size)
                    self.update_rec_matrix(user, item, pred_rating)

    def compute_mae(self) -> float:
        """ Compute the mean absolute error of the prediction
        @return: the mean absolute error
        """
        numerator = 0
        denominator = 0
        for user in range(self.num_users):
            for item in range(self.num_items):
                if (self.matrix[user][item] == self.no_review):
                    continue
                numerator += abs(self.rec_matrix[user]
                                 [item] - self.matrix[user][item])
                denominator += 1
        mae = numerator / denominator
        return mae

    def print_rec_matrix(self) -> None:
        """ Print the recommendation matrix
        """
        for row in self.rec_matrix:
            print(row)


if __name__ == "__main__":
    # max_threads = 4
    # neighborhood_size = 2
    # no_review = -1
    # filename = "test3.txt"
    # cross_validation = False

    max_threads = 1
    neighborhood_size = 5
    no_review = 0 
    filename = "assignment2-data.txt"
    # filename = "assignment2-data.txt"
    cross_validation = True

    print("Initializing RecommenderCrossValidator...")
    validator = RecommenderCrossValidator(filename, max_threads, no_review)
    row = validator.num_users
    col = validator.num_items
    print(
        f"Done initializing RecommenderCrossValidator, row: {row}, col: {col}")

    print("Starting pred matrix calculation")
    threads = []
    for thread_id in range(max_threads):
        start_row = int(thread_id * row / max_threads)
        end_row = int((thread_id + 1) * row / max_threads)
        thread = Thread(target=validator.update_pred,
                        args=(thread_id, start_row, end_row, col, neighborhood_size, cross_validation))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print("Finished pred matrix calculation")

    if (not cross_validation):
        validator.print_rec_matrix()

    print("Computing MAE")
    mae = validator.compute_mae()
    print(f"MAE: {mae}")
