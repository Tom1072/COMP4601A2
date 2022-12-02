import time
import math
import json
import numpy as np
from item_based_rec import ItemBasedRecommender
from user_based_rec import UserBasedRecommender
from threading import Thread, Lock
from tqdm import tqdm

INPUT_FILE = "assignment2-data.txt"
OUTPUT_FILE = "assignment2-results.json"


class RecommenderCrossValidator():
    def __init__(self, filename, result_filename, no_review=0) -> None:
        self.results = {
            "item-based": {
                "neighborhood-size": [
                    # {
                    #     "neighborhood-size": 0,
                    #     "mae": 0,
                    #     "time-in-seconds": 0
                    # }
                ],
                "sim-threshold": [
                    # {
                    #     "sim-threshold": 0,
                    #     "mae": 0,
                    #     "time-in-seconds": 0
                    # }
                ],
                "absolute-sim-threshold": [
                    # {
                    #     "absolute-sim-threshold": 0,
                    #     "mae": 0,
                    #     "time-in-seconds": 0
                    # }
                ]
            },
            "user-based": {
                "neighborhood-size": [
                    # {
                    #     "neighborhood-size": 0,
                    #     "mae": 0,
                    #     "time-in-seconds": 0
                    # }
                ],
                "sim-threshold": [
                    # {
                    #     "sim-threshold": 0,
                    #     "mae": 0,
                    #     "time-in-seconds": 0
                    # }
                ],
                "absolute-sim-threshold": [
                    # {
                    #     "absolute-sim-threshold": 0,
                    #     "mae": 0,
                    #     "time-in-seconds": 0
                    # }
                ]
            }
        }
        self.lock = Lock()
        self.result_file = open(result_filename, "w")
        self.item_based_rec = ItemBasedRecommender(filename, no_review)
        self.user_based_rec = UserBasedRecommender(filename, no_review)

        self.no_review = no_review
        self.num_users = self.item_based_rec.num_users
        self.num_items = self.item_based_rec.num_items
        self.matrix = self.item_based_rec.matrix

        self.item_based_neighborhood_size_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.item_based_sim_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.item_based_absolute_sim_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.user_based_neighborhood_size_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.user_based_sim_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.user_based_absolute_sim_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.result_file.close()

    def boundary_value(self, value: float) -> float:
        if value < 1:
            return 1
        if value > 5:
            return 5
        return value

    def compute_mae(self, pred_matrix) -> float:
        """ Compute the mean absolute error of the prediction
        @return: the mean absolute error
        """
        numerator = 0
        denominator = 0

        for user in range(self.num_users):
            for item in range(self.num_items):
                if (self.matrix[user][item] == self.no_review):
                    continue
                numerator += abs(pred_matrix[user]
                                 [item] - self.matrix[user][item])
                denominator += 1

        mae = numerator / denominator

        return mae

    def item_based_validate(self, start_neighborhood_size=1, num_of_neighborhood_steps=10, start_sim_threshold=-1, end_sim_threshold=1, start_abs_sim_threshold=0, end_abs_sim_threshold=1) -> None:
        sim_threshold_step = 0.1
        total_iterations = (num_of_neighborhood_steps + round((end_sim_threshold - start_sim_threshold) / sim_threshold_step) +
                            round((end_abs_sim_threshold - start_abs_sim_threshold) / sim_threshold_step)) * self.num_items * self.num_users

        print("Total iterations:", total_iterations)
        print("num_items:", self.num_items)
        print("num_users:", self.num_users)
        progress_bar = tqdm(total=total_iterations,
                            desc="Item-based validation")
        for step in range(num_of_neighborhood_steps):
            neighborhood_size = start_neighborhood_size + step * 2
            neighborhood_size_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.item_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    neighborhood_size_pred_rating = self.item_based_rec.pred_with_neighborhood_size(
                        user, item, neighborhood_size)
                    end_time = time.time()
                    neighborhood_size_time += end_time - start_time
                    self.item_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

                    self.item_based_neighborhood_size_rec_matrix[user][item] = self.boundary_value(
                        neighborhood_size_pred_rating)

            neighborhood_size_mae = self.compute_mae(
                self.item_based_neighborhood_size_rec_matrix)
            # print(f"Item-based {neighborhood_size=}, {neighborhood_size_mae=}, {neighborhood_size_time=}")

            self.lock.acquire()
            self.results["item-based"]["neighborhood-size"].append({
                "neighborhood-size": neighborhood_size,
                "mae": neighborhood_size_mae,
                "time": neighborhood_size_time
            })
            self.lock.release()

        for sim_threshold in np.arange(start_sim_threshold, end_sim_threshold, sim_threshold_step):
            sim_threshold_time = 0
            absolute_sim_threshold_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.item_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    sim_threshold_pred_rating = self.item_based_rec.pred_with_sim_threshold(
                        user, item, sim_threshold)
                    end_time = time.time()
                    sim_threshold_time += end_time - start_time
                    self.item_based_sim_threshold_rec_matrix[user][item] = self.boundary_value(
                        sim_threshold_pred_rating)

                    self.item_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            sim_threshold_mae = self.compute_mae(
                self.item_based_sim_threshold_rec_matrix)
            # print(f"Item-based {sim_threshold=}, {sim_threshold_mae=}, {sim_threshold_time=}")
            self.lock.acquire()
            self.results["item-based"]["sim-threshold"].append({
                "sim-threshold": sim_threshold,
                "mae": sim_threshold_mae,
                "time": sim_threshold_time
            })
            self.lock.release()

        for sim_threshold in np.arange(start_abs_sim_threshold, end_abs_sim_threshold, sim_threshold_step):
            sim_threshold_time = 0
            absolute_sim_threshold_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.item_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    absolute_sim_threshold_pred_rating = self.item_based_rec.pred_with_absolute_sim_threshold(
                        user, item, sim_threshold)
                    end_time = time.time()
                    absolute_sim_threshold_time += end_time - start_time
                    self.item_based_absolute_sim_threshold_rec_matrix[user][item] = self.boundary_value(
                        absolute_sim_threshold_pred_rating)

                    self.item_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            absolute_sim_threshold = sim_threshold
            absolute_sim_threshold_mae = self.compute_mae(
                self.item_based_absolute_sim_threshold_rec_matrix)
            # print(f"Item-based {absolute_sim_threshold=}, {absolute_sim_threshold_mae=}, {absolute_sim_threshold_time=}")
            self.lock.acquire()
            self.results["item-based"]["absolute-sim-threshold"].append({
                "absolute-sim-threshold": absolute_sim_threshold,
                "mae": absolute_sim_threshold_mae,
                "time": absolute_sim_threshold_time
            })
            self.lock.release()
        progress_bar.close()

    def user_based_validate(self, start_neighborhood_size=1, num_of_neighborhood_steps=10, start_sim_threshold=-1, end_sim_threshold=1, start_abs_sim_threshold=0, end_abs_sim_threshold=1) -> None:
        sim_threshold_step = 0.1
        total_iterations = (num_of_neighborhood_steps + round((end_sim_threshold - start_sim_threshold) / sim_threshold_step) +
                            round((end_abs_sim_threshold - start_abs_sim_threshold) / sim_threshold_step)) * self.num_items * self.num_users
        
        print("Total iterations:", total_iterations)
        print("num_items:", self.num_items)
        print("num_users:", self.num_users)

        progress_bar = tqdm(total=total_iterations,
                            desc="User-based validation")
        for step in range(num_of_neighborhood_steps):
            neighborhood_size = start_neighborhood_size + step * 2
            neighborhood_size_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.user_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    neighborhood_size_pred_rating = self.user_based_rec.pred_with_neighborhood_size(
                        user, item, neighborhood_size)
                    end_time = time.time()
                    neighborhood_size_time += end_time - start_time
                    self.user_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

                    self.user_based_neighborhood_size_rec_matrix[user][item] = self.boundary_value(
                        neighborhood_size_pred_rating)

            neighborhood_size_mae = self.compute_mae(
                self.user_based_neighborhood_size_rec_matrix)
            # print(f"User-based {neighborhood_size=}, {neighborhood_size_mae=}, {neighborhood_size_time=}")
            self.lock.acquire()
            self.results["user-based"]["neighborhood-size"].append({
                "neighborhood-size": neighborhood_size,
                "mae": neighborhood_size_mae,
                "time": neighborhood_size_time
            })
            self.lock.release()

        for sim_threshold in np.arange(start_sim_threshold, end_sim_threshold, sim_threshold_step):
            sim_threshold_time = 0
            absolute_sim_threshold_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.user_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    sim_threshold_pred_rating = self.user_based_rec.pred_with_sim_threshold(
                        user, item, sim_threshold)
                    end_time = time.time()
                    sim_threshold_time += end_time - start_time
                    self.user_based_sim_threshold_rec_matrix[user][item] = self.boundary_value(
                        sim_threshold_pred_rating)

                    self.user_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            sim_threshold_mae = self.compute_mae(
                self.user_based_sim_threshold_rec_matrix)
            # print(f"User-based {sim_threshold=}, {sim_threshold_mae=}, {sim_threshold_time=}")
            self.lock.acquire()
            self.results["user-based"]["sim-threshold"].append({
                "sim-threshold": sim_threshold,
                "mae": sim_threshold_mae,
                "time": sim_threshold_time
            })
            self.lock.release()

        for absolute_sim_threshold in np.arange(start_abs_sim_threshold, end_abs_sim_threshold, sim_threshold_step):
            sim_threshold_time = 0
            absolute_sim_threshold_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.user_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    absolute_sim_threshold_pred_rating = self.user_based_rec.pred_with_absolute_sim_threshold(
                        user, item, sim_threshold)
                    end_time = time.time()
                    absolute_sim_threshold_time += end_time - start_time
                    self.user_based_absolute_sim_threshold_rec_matrix[user][item] = self.boundary_value(
                        absolute_sim_threshold_pred_rating)

                    self.user_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            absolute_sim_threshold_mae = self.compute_mae(
                self.user_based_absolute_sim_threshold_rec_matrix)
            # print(f"User-based {absolute_sim_threshold=}, {absolute_sim_threshold_mae=}, {absolute_sim_threshold_time=}")
            self.lock.acquire()
            self.results["user-based"]["absolute-sim-threshold"].append({
                "absolute-sim-threshold": absolute_sim_threshold,
                "mae": absolute_sim_threshold_mae,
                "time": absolute_sim_threshold_time
            })
            self.lock.release()
        progress_bar.close()

    def start_validation(self):
        # item_based_thread = Thread(
        #     target=self.item_based_validate, args=(5, 1, -1, -0.9, -1, -0.9))
        # user_based_thread = Thread(
        #     target=self.user_based_validate, args=(5, 1, -1, -0.9, -1, -0.9))

        item_based_thread = Thread(target=self.item_based_validate)
        user_based_thread = Thread(target=self.user_based_validate)

        item_based_thread.start()
        user_based_thread.start()

        item_based_thread.join()
        user_based_thread.join()

        print("Done validation, saving results")
        self.result_file.write(json.dumps(self.results))


if __name__ == "__main__":
    filename = "parsed-data-trimmed.txt"
    result_filename = "parsed-data-trimmed-results.json"

    with RecommenderCrossValidator(INPUT_FILE, OUTPUT_FILE) as validator:
        validator.start_validation()
