from item_based_rec import ItemBasedRecommender
from user_based_rec import UserBasedRecommender
from threading import Thread, Lock
from tqdm import tqdm
    
class RecommenderCrossValidator():
    def __init__(self, filename, num_threads, no_review=0) -> None:
        self.item_based_rec = ItemBasedRecommender(filename, num_threads, no_review)
        self.user_based_rec = UserBasedRecommender(filename, num_threads, no_review)

        self.no_review = no_review
        self.num_users = self.item_based_rec.num_users
        self.num_items = self.item_based_rec.num_items
        self.matrix = self.item_based_rec.matrix

        self.item_based_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]
        self.item_based_rec_matrix_lock = Lock()

        self.user_based_rec_matrix = [[0 for _ in range(
            self.num_items)] for _ in range(self.num_users)]
        self.user_based_rec_matrix_lock = Lock()

    def update_rec_matrix(self, user: int, item: int, item_based_pred_rating: float, user_based_pred_rating) -> None:
        item_based_pred_rating = 5 if item_based_pred_rating > 5 else item_based_pred_rating
        item_based_pred_rating = 1 if item_based_pred_rating < 1 else item_based_pred_rating
        user_based_pred_rating = 5 if user_based_pred_rating > 5 else user_based_pred_rating
        user_based_pred_rating = 1 if user_based_pred_rating < 1 else user_based_pred_rating

        self.item_based_rec_matrix_lock.acquire()
        self.item_based_rec_matrix[user][item] = item_based_pred_rating
        self.item_based_rec_matrix_lock.release()

        self.user_based_rec_matrix_lock.acquire()
        self.user_based_rec_matrix[user][item] = user_based_pred_rating
        self.user_based_rec_matrix_lock.release()

    def update_pred(self, thread_id: int, start_row: int, end_row: int, col: int, neighborhood_size: int, sim_threshold: float, cross_val: bool = True) -> None:
        progress_bar = tqdm(total=(end_row - start_row) * self.num_items)
        progress_bar.set_description(f"Pred Thread {thread_id:2.0f}")
        # for user in tqdm(range(start_row, end_row)):
        for user in range(start_row, end_row):
            for item in range(col):
                progress_bar.update(1)
                if cross_val:
                    if self.matrix[user][item] == no_review:
                        continue

                    self.item_based_rec.update_ratings(thread_id, user, item, self.no_review)
                    item_based_pred_rating = self.item_based_rec.pred(thread_id, user, item, neighborhood_size, sim_threshold)
                    self.item_based_rec.update_ratings(
                        thread_id, user, item, self.matrix[user][item])
                    
                    self.user_based_rec.update_ratings(thread_id, user, item, self.no_review)
                    user_based_pred_rating = self.user_based_rec.pred(thread_id, user, item, neighborhood_size, sim_threshold)
                    self.user_based_rec.update_ratings(
                        thread_id, user, item, self.matrix[user][item])

                    self.update_rec_matrix(user, item, item_based_pred_rating, user_based_pred_rating)
                else:
                    if self.matrix[user][item] != no_review:
                        self.update_rec_matrix(user, item, self.matrix[user][item], self.matrix[user][item])
                        continue
                    item_based_pred_rating = self.item_based_rec.pred(
                        thread_id, user, item, neighborhood_size)
                    user_based_pred_rating = self.user_based_rec.pred(
                        thread_id, user, item, neighborhood_size)
                    self.update_rec_matrix(user, item, item_based_pred_rating, user_based_pred_rating)

    def compute_mae(self) -> float:
        """ Compute the mean absolute error of the prediction
        @return: the mean absolute error
        """
        item_based_numerator = 0
        user_based_numerator = 0
        denominator = 0
        for user in range(self.num_users):
            for item in range(self.num_items):
                if (self.matrix[user][item] == self.no_review):
                    continue
                item_based_numerator += abs(self.item_based_rec_matrix[user][item] - self.matrix[user][item])
                user_based_numerator += abs(self.user_based_rec_matrix[user][item] - self.matrix[user][item])
                denominator += 1
        item_based_mae = item_based_numerator / denominator
        user_based_mae = user_based_numerator / denominator
        return item_based_mae, user_based_mae

    def print_rec_matrix(self) -> None:
        """ Print the recommendation matrix
        """
        print("Item-based Recommendation Matrix")
        for row in self.item_based_rec_matrix:
            print(row)

        print()

        print("User-based Recommendation Matrix")
        for row in self.user_based_rec_matrix:
            print(row)


if __name__ == "__main__":
    # max_threads = 1
    # neighborhood_size = 2
    # no_review = -1
    # filename = "lab6/test3.txt"
    # cross_validation = False
    # sim_threshold = -1 # no threshold

    max_threads = 4
    neighborhood_size = 5
    no_review = 0 
    # filename = "assignment2-data.txt"
    filename = "parsed-data-trimmed.txt"
    cross_validation = True
    sim_threshold = -1

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
                        args=(thread_id, start_row, end_row, col, neighborhood_size, sim_threshold, cross_validation))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print("Finished pred matrix calculation")

    if (not cross_validation):
        validator.print_rec_matrix()

    print("Computing MAE")
    item_based_mae, user_based_mae = validator.compute_mae()
    print(f"Item-based MAE: {item_based_mae}")
    print(f"User-based MAE: {user_based_mae}")
