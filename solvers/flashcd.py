from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    from flashcd import MCP


class Solver(BaseSolver):
    name = "flashcd"

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-4, patience=50, strategy="iteration"
    )

    def set_objective(self, X, y, lmbd, gamma):
        # self.X, self.y = np.asfortranarray(X), y
        self.X, self.y = X, y
        self.lmbd, self.gamma = lmbd, gamma
        self.clf = MCP(alpha=lmbd, gamma=gamma,
                       fit_intercept=False)

        # Make sure we cache the numba compilation.
        self.run(1)
        self.clf.tol = 1e-12

    def run(self, n_iter):
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    @staticmethod
    def get_next(stop_val):
        return stop_val + 1

    def get_result(self):
        return self.clf.coef_
