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
        self.result_filename = result_filename
        try:
            with open(result_filename, "r") as f:
                self.results = json.load(f)
        except:
            self.results = {}
        if "item-based" not in self.results:
            self.results.update({"item-based": {"neighborhood-size": [],
                                "sim-threshold": [], "absolute-sim-threshold": []}})
        if "neighborhood-size" not in self.results["item-based"]:
            self.results["item-based"].update({"neighborhood-size": []})
        if "sim-threshold" not in self.results["item-based"]:
            self.results["item-based"].update({"sim-threshold": []})
        if "absolute-sim-threshold" not in self.results["item-based"]:
            self.results["item-based"].update({"absolute-sim-threshold": []})
        if "size-and-threshold" not in self.results["item-based"]:
            self.results["item-based"].update({"size-and-threshold": []})
        if "size-and-absolute-threshold" not in self.results["item-based"]:
            self.results["item-based"].update(
                {"size-and-absolute-threshold": []})

        if "user-based" not in self.results:
            self.results.update({"user-based": {"neighborhood-size": [],
                                "sim-threshold": [], "absolute-sim-threshold": []}})
        if "neighborhood-size" not in self.results["user-based"]:
            self.results["user-based"].update({"neighborhood-size": []})
        if "sim-threshold" not in self.results["user-based"]:
            self.results["user-based"].update({"sim-threshold": []})
        if "absolute-sim-threshold" not in self.results["user-based"]:
            self.results["user-based"].update({"absolute-sim-threshold": []})
        if "size-and-threshold" not in self.results["user-based"]:
            self.results["user-based"].update({"size-and-threshold": []})
        if "size-and-absolute-threshold" not in self.results["user-based"]:
            self.results["user-based"].update(
                {"size-and-absolute-threshold": []})

        # print(json.dumps(self.results))

        """
        self.results = {
            "item-based": {
                "neighborhood-size": [
                    {
                        "neighborhood-size": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ],
                "sim-threshold": [
                    {
                        "sim-threshold": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ],
                "absolute-sim-threshold": [
                    {
                        "absolute-sim-threshold": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ],
                "size-and-threshold": [
                    {
                        "neighborhood-size": 0,
                        "sim-threshold": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ],
                "size-and-absolute-threshold": [
                    {
                        "neighborhood-size": 0,
                        "absolute-sim-threshold": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ]
            },
            "user-based": {
                "neighborhood-size": [
                    {
                        "neighborhood-size": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ],
                "sim-threshold": [
                    {
                        "sim-threshold": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ],
                "absolute-sim-threshold": [
                    {
                        "absolute-sim-threshold": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ],
                "size-and-threshold": [
                    {
                        "neighborhood-size": 0,
                        "sim-threshold": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ],
                "size-and-absolute-threshold": [
                    {
                        "neighborhood-size": 0,
                        "absolute-sim-threshold": 0,
                        "mae": 0,
                        "time-in-seconds": 0
                    }
                ]
            }
        }
        """
        self.lock = Lock()
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

        self.item_based_size_and_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.item_based_size_and_absolute_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.user_based_neighborhood_size_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.user_based_sim_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.user_based_absolute_sim_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.user_based_size_and_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

        self.user_based_size_and_absolute_threshold_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

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

    def item_based_validate(self, neighborhood_sizes: list = [], sim_thresholds: list = [], absolute_sim_thresholds: list = [], size_and_thresholds: list = [], size_and_absolute_thresholds: list = []) -> None:
        total_iterations = (len(neighborhood_sizes) + len(sim_thresholds) + len(absolute_sim_thresholds) + len(
            size_and_thresholds) + len(size_and_absolute_thresholds)) * self.num_users * self.num_items

        print("Total iterations:", total_iterations)
        progress_bar = tqdm(total=total_iterations,
                            desc="Item-based validation")
        for size in neighborhood_sizes:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.item_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.item_based_rec.pred_with_neighborhood_size(
                        user, item, size)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.item_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

                    self.item_based_neighborhood_size_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

            mae = self.compute_mae(
                self.item_based_neighborhood_size_rec_matrix)


            self.lock.acquire()
            self.results["item-based"]["neighborhood-size"].append({
                "neighborhood-size": size,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()

        for threshold in sim_thresholds:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.item_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.item_based_rec.pred_with_sim_threshold(
                        user, item, threshold)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.item_based_sim_threshold_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

                    self.item_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            mae = self.compute_mae(
                self.item_based_sim_threshold_rec_matrix)

            self.lock.acquire()
            self.results["item-based"]["sim-threshold"].append({
                "sim-threshold": threshold,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()

        for threshold in absolute_sim_thresholds:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.item_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.item_based_rec.pred_with_absolute_sim_threshold(
                        user, item, threshold)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.item_based_absolute_sim_threshold_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

                    self.item_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            mae = self.compute_mae(
                self.item_based_absolute_sim_threshold_rec_matrix)

            self.lock.acquire()
            self.results["item-based"]["absolute-sim-threshold"].append({
                "absolute-sim-threshold": threshold,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()

        for size, threshold in size_and_thresholds:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.item_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.item_based_rec.pred_with_size_and_threshold(
                        user, item, size, threshold)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.item_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

                    self.item_based_size_and_threshold_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

            mae = self.compute_mae(
                self.item_based_size_and_threshold_rec_matrix)

            self.lock.acquire()
            self.results["item-based"]["size-and-threshold"].append({
                "neighborhood-size": size,
                "sim-threshold": threshold,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()

        for size, threshold in size_and_absolute_thresholds:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.item_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.item_based_rec.pred_with_size_and_absolute_threshold(
                        user, item, size, threshold)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.item_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

                    self.item_based_size_and_absolute_threshold_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

            mae = self.compute_mae(
                self.item_based_size_and_absolute_threshold_rec_matrix)


            self.lock.acquire()
            self.results["item-based"]["size-and-absolute-threshold"].append({
                "neighborhood-size": size,
                "absolute-sim-threshold": threshold,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()
        progress_bar.close()

    def user_based_validate(self, neighborhood_sizes: list = [], sim_thresholds: list = [], absolute_sim_thresholds: list = [], size_and_thresholds: list = [], size_and_absolute_thresholds: list = []) -> None:
        total_iterations = (len(neighborhood_sizes) + len(sim_thresholds) + len(absolute_sim_thresholds) + len(size_and_thresholds) + len(size_and_absolute_thresholds)) * self.num_items * self.num_users

        print("Total iterations:", total_iterations)

        progress_bar = tqdm(total=total_iterations,
                            desc="User-based validation")
        for size in neighborhood_sizes:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.user_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.user_based_rec.pred_with_neighborhood_size(
                        user, item, size)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.user_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

                    self.user_based_neighborhood_size_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

            mae = self.compute_mae(
                self.user_based_neighborhood_size_rec_matrix)
            # print(f"User-based {neighborhood_size=}, {neighborhood_size_mae=}, {neighborhood_size_time=}")
            self.lock.acquire()
            self.results["user-based"]["neighborhood-size"].append({
                "neighborhood-size": size,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()

        for threshold in sim_thresholds:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.user_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.user_based_rec.pred_with_sim_threshold(
                        user, item, threshold)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.user_based_sim_threshold_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

                    self.user_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            mae = self.compute_mae(
                self.user_based_sim_threshold_rec_matrix)
            # print(f"User-based {sim_threshold=}, {sim_threshold_mae=}, {sim_threshold_time=}")
            self.lock.acquire()
            self.results["user-based"]["sim-threshold"].append({
                "sim-threshold": threshold,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()

        for threshold in absolute_sim_thresholds:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.user_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.user_based_rec.pred_with_absolute_sim_threshold(
                        user, item, threshold)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.user_based_absolute_sim_threshold_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

                    self.user_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            mae = self.compute_mae(
                self.user_based_absolute_sim_threshold_rec_matrix)
            # print(f"User-based {absolute_sim_threshold=}, {absolute_sim_threshold_mae=}, {absolute_sim_threshold_time=}")
            self.lock.acquire()
            self.results["user-based"]["absolute-sim-threshold"].append({
                "absolute-sim-threshold": threshold,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()

        for size, threshold in size_and_thresholds:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.user_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.user_based_rec.pred_with_size_and_threshold(
                        user, item, size, threshold)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.user_based_size_and_threshold_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

                    self.user_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            mae = self.compute_mae(
                self.user_based_size_and_threshold_rec_matrix)
            self.lock.acquire()
            self.results["user-based"]["size-and-threshold"].append({
                "neighborhood-size": size,
                "sim-threshold": threshold,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()

        for size, threshold in size_and_absolute_thresholds:
            total_time = 0
            for user in range(self.num_users):
                for item in range(self.num_items):
                    progress_bar.update(1)
                    if self.matrix[user][item] == self.no_review:
                        continue

                    self.user_based_rec.update_ratings(
                        user, item, self.no_review)

                    start_time = time.time()
                    pred_rating = self.user_based_rec.pred_with_size_and_absolute_threshold(
                        user, item, size, threshold)
                    end_time = time.time()
                    total_time += end_time - start_time
                    self.user_based_size_and_absolute_threshold_rec_matrix[user][item] = self.boundary_value(
                        pred_rating)

                    self.user_based_rec.update_ratings(
                        user, item, self.matrix[user][item])

            mae = self.compute_mae(
                self.user_based_size_and_absolute_threshold_rec_matrix)
            self.lock.acquire()
            self.results["user-based"]["size-and-absolute-threshold"].append({
                "neighborhood-size": size,
                "absolute-sim-threshold": threshold,
                "mae": mae,
                "time": total_time
            })
            self.lock.release()
        progress_bar.close()

    def start_validation(self):
        print(f"num_users: {self.num_users}, num_items: {self.num_items}")

        # Exponential step
        step_exponent = 2
        item_neighborhood_sizes = [
            0] + [round(pow(step_exponent, i)) for i in range(0, math.ceil(math.log(self.num_items, step_exponent)))]
        user_neighborhood_sizes = [
            0] + [round(pow(step_exponent, i)) for i in range(0, math.ceil(math.log(self.num_users, step_exponent)))]
        sim_thresholds = [threshold/10 for threshold in range(-10, 11, 1)]
        absolute_sim_thresholds = [threshold / 10 for threshold in range(0, 11, 1)]

        # Linear step
        # user_neighborhood_sizes = [i for i in range(0, self.num_users + 1)]
        # item_neighborhood_sizes = [i for i in range(0, self.num_items + 1)]
        # sim_thresholds = [threshold/100 for threshold in range(-100, 101, 1)]
        # absolute_sim_thresholds = [threshold /
        #                            100 for threshold in range(0, 101, 1)]
        item_size_and_threshold = [
            (size, threshold) for size in item_neighborhood_sizes for threshold in sim_thresholds]
        item_size_and_absolute_threshold = [
            (size, threshold) for size in item_neighborhood_sizes for threshold in absolute_sim_thresholds]
        user_size_and_threshold = [
            (size, threshold) for size in user_neighborhood_sizes for threshold in sim_thresholds]
        user_size_and_absolute_threshold = [
            (size, threshold) for size in user_neighborhood_sizes for threshold in absolute_sim_thresholds]

        # Test
        # item_size_and_threshold = [(5, -1)]
        # item_size_and_absolute_threshold = [(5, 0)]
        # user_size_and_threshold = [(5, -1)]
        # user_size_and_absolute_threshold = [(5, 0)]


        # item_based_thread = Thread(target=self.item_based_validate, args=(
        #     item_neighborhood_sizes, sim_thresholds, absolute_sim_thresholds, item_size_and_threshold, item_size_and_absolute_threshold))
        # user_based_thread = Thread(target=self.user_based_validate, args=(
        #     user_neighborhood_sizes, sim_thresholds, absolute_sim_thresholds, user_size_and_threshold, user_size_and_absolute_threshold))
        item_based_thread = Thread(target=self.item_based_validate, args=(
            [], [], [], item_size_and_threshold, item_size_and_absolute_threshold))
        user_based_thread = Thread(target=self.user_based_validate, args=(
            [], [], [], user_size_and_threshold, user_size_and_absolute_threshold))

        item_based_thread.start()
        user_based_thread.start()

        item_based_thread.join()
        user_based_thread.join()

        print("Done validation, saving results")
        with open(self.result_filename, "w") as f:
            json.dump(self.results, f, indent=4)


if __name__ == "__main__":
    # with RecommenderCrossValidator(TEST_INPUT_FILE, TEST_OUTPUT_FILE) as validator:
    #     validator.start_validation()
    with RecommenderCrossValidator(INPUT_FILE, OUTPUT_FILE) as validator:
        validator.start_validation()
