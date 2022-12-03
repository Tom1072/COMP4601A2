import time
import json
import math
from item_based_rec import ItemBasedRecommender
from user_based_rec import UserBasedRecommender
from threading import Thread, Lock
from tqdm import tqdm

INPUT_FILE = "assignment2-data.txt"
OUTPUT_FILE = "assignment2-results.json"

TEST_INPUT_FILE = "parsed-data-trimmed.txt"
TEST_OUTPUT_FILE = "parsed-data-trimmed-results.json"


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

    def item_based_validate(self, neighborhood_sizes: list, sim_thresholds: list, absolute_sim_thresholds:list) -> None:
        total_iterations = (len(neighborhood_sizes) + len(sim_thresholds) + len(absolute_sim_thresholds)) * self.num_users * self.num_items

        print("Total iterations:", total_iterations)
        progress_bar = tqdm(total=total_iterations,
                            desc="Item-based validation")
        for neighborhood_size in neighborhood_sizes:
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

        for sim_threshold in sim_thresholds:
            sim_threshold_time = 0
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

        for absolute_sim_threshold in absolute_sim_thresholds:
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
                        user, item, absolute_sim_threshold)
                    end_time = time.time()
                    absolute_sim_threshold_time += end_time - start_time
                    self.item_based_absolute_sim_threshold_rec_matrix[user][item] = self.boundary_value(
                        absolute_sim_threshold_pred_rating)

                    self.item_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

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

    def user_based_validate(self, neighborhood_sizes: list, sim_thresholds: list, absolute_sim_thresholds:list) -> None:
        total_iterations = (len(neighborhood_sizes) + len(sim_thresholds) + len(absolute_sim_thresholds)) * self.num_items * self.num_users

        print("Total iterations:", total_iterations)

        progress_bar = tqdm(total=total_iterations,
                            desc="User-based validation")
        for neighborhood_size in neighborhood_sizes:
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

        for sim_threshold in sim_thresholds:
            sim_threshold_time = 0
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

        for absolute_sim_threshold in absolute_sim_thresholds:
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
                        user, item, absolute_sim_threshold)
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
        print(f"num_users: {self.num_users}, num_items: {self.num_items}")



        # item_neighborhood_sizes = [0] + [pow(2, i) for i in range(0, math.ceil(math.log(self.num_items, 2)))]
        # user_neighborhood_sizes = [0] + [pow(2, i) for i in range(0, math.ceil(math.log(self.num_users, 2)))]
        user_neighborhood_sizes = [i for i in range(0, self.num_users + 1)]
        item_neighborhood_sizes = [i for i in range(0, self.num_items + 1)]
        sim_thresholds = [threshold/100 for threshold in range(-100, 101, 1)]
        absolute_sim_thresholds = [threshold/100 for threshold in range(0, 101, 1)]

        # neighborhood_sizes = [5]
        # sim_thresholds = [-1]
        # absolute_sim_thresholds = [0]

        print(f"{item_neighborhood_sizes=}")
        print(f"{user_neighborhood_sizes=}")
        print(f"{sim_thresholds=}")
        print(f"{absolute_sim_thresholds=}")

        item_based_thread = Thread(target=self.item_based_validate, args=(item_neighborhood_sizes, sim_thresholds, absolute_sim_thresholds))
        user_based_thread = Thread(target=self.user_based_validate, args=(user_neighborhood_sizes, sim_thresholds, absolute_sim_thresholds))

        item_based_thread.start()
        user_based_thread.start()

        item_based_thread.join()
        user_based_thread.join()

        print("Done validation, saving results")
        self.result_file.write(json.dumps(self.results))


if __name__ == "__main__":
    # with RecommenderCrossValidator(TEST_INPUT_FILE, TEST_OUTPUT_FILE) as validator:
    #     validator.start_validation()
    with RecommenderCrossValidator(INPUT_FILE, OUTPUT_FILE) as validator:
        validator.start_validation()
