import numpy as np
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles

sns.set()


class SVM:
    """
        Class for main
    """

    NUMBER_OF_PARTITIONS = 2
    FIG_SIZE = (16, 6,)

    def run_tests_blobs(self) -> None:
        """
            Method to run tests in blobs !
        """

        # test cases
        tests = [
            dict(N=100, random_state=0, cluster_std=0.5, gamma='auto'),
            dict(N=400, random_state=5, cluster_std=0.4, gamma='scale'),
            dict(N=100, random_state=0, cluster_std=0.5, gamma='auto'),
            dict(N=400, random_state=5, cluster_std=0.4, gamma='scale'),
            dict(N=100, random_state=0, cluster_std=0.5, kernel="poly", degree=70),
            dict(N=100, random_state=0, cluster_std=0.5, kernel="poly", degree=3),
            dict(N=400, random_state=5, cluster_std=0.4, kernel="poly", degree=6),
            dict(N=400, random_state=5, cluster_std=0.4, kernel="poly", degree=40),
            dict(N=100, random_state=0, cluster_std=0.5, kernel="linear"),
            dict(N=400, random_state=5, cluster_std=0.4, kernel="linear"),
            dict(N=100, random_state=0, cluster_std=0.5, kernel="sigmoid"),
            dict(N=400, random_state=5, cluster_std=0.4, kernel="sigmoid"),
        ]

        for i in range(0, len(tests), 2):
            # make the subplot
            fig, ax = plt.subplots(1, 2, figsize=self.FIG_SIZE)
            fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

            # fill subplots
            self.plot_svm1(ax[0], **tests[i])
            self.plot_svm1(ax[1], **tests[i + 1])

            ax[0].set_title(f'N = {tests[i]["N"]}')
            ax[1].set_title(f'N = {tests[i + 1]["N"]}')

            # show plot
            plt.show()

    def run_tests_circle(self) -> None:
        """
            Method to run tests in circle !
        """

        # test cases
        tests = [
            dict(N=100, random_state=0, factor=0.6, noise=0.1, gamma='auto'),
            dict(N=100, random_state=10, factor=0.3, noise=0.1, gamma='auto', shuffle=True),
            dict(N=400, random_state=0, factor=0.6, noise=0.1, gamma='scale'),
            dict(N=400, random_state=10, factor=0.3, noise=0.1, gamma='scale', shuffle=True),
            dict(N=100, random_state=0, factor=0.6, noise=0.1, kernel="poly", degree=70),
            dict(N=100, random_state=10, factor=0.3, noise=0.1, kernel="poly", degree=3),
            dict(N=400, random_state=0, factor=0.6, noise=0.1, kernel="poly", degree=6),
            dict(N=400, random_state=10, factor=0.3, noise=0.1, kernel="poly", degree=40),
            dict(N=100, random_state=0, factor=0.6, noise=0.1, kernel="linear", shuffle=True),
            dict(N=100, random_state=10, factor=0.3, noise=0.1, kernel="linear"),
            dict(N=400, random_state=0, factor=0.6, noise=0.1, kernel="sigmoid", shuffle=True),
            dict(N=400, random_state=10, factor=0.3, noise=0.1, kernel="sigmoid"),

            dict(N=100, random_state=0, factor=0.3, noise=0.3, gamma='auto'),
            dict(N=100, random_state=10, factor=0.3, noise=0.5, gamma='auto', shuffle=True),
            dict(N=400, random_state=0, factor=0.3, noise=0.3, gamma='scale'),
            dict(N=400, random_state=10, factor=0.3, noise=0.5, gamma='scale', shuffle=True),
            dict(N=100, random_state=0, factor=0.3, noise=0.3, kernel="poly", degree=70),
            dict(N=100, random_state=10, factor=0.3, noise=0.5, kernel="poly", degree=3),
            dict(N=400, random_state=0, factor=0.3, noise=0.3, kernel="poly", degree=6),
            dict(N=400, random_state=10, factor=0.3, noise=0.5, kernel="poly", degree=40),
            dict(N=100, random_state=0, factor=0.3, noise=0.3, kernel="linear", shuffle=True),
            dict(N=100, random_state=10, factor=0.3, noise=0.5, kernel="linear"),
            dict(N=400, random_state=0, factor=0.3, noise=0.3, kernel="sigmoid", shuffle=True),
            dict(N=400, random_state=10, factor=0.3, noise=0.5, kernel="sigmoid"),
        ]

        for i in range(0, len(tests), 2):
            # make the subplot
            fig, ax = plt.subplots(1, 2, figsize=self.FIG_SIZE)
            fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

            # fill subplots
            self.plot_svm2(ax[0], **tests[i])
            self.plot_svm2(ax[1], **tests[i + 1])

            ax[0].set_title(f'N = {tests[i]["N"]}')
            ax[1].set_title(f'N = {tests[i + 1]["N"]}')

            # show plot
            plt.show()

    @staticmethod
    def plot_svc_decision_function(model: SVC, axes=None, plot_support: bool = True):
        """
            Plot the decision function for a 2D SVC,

            Parameters:
                model (SVC): The model
                axes: The axes
                plot_support (bool): The support for circle or blob
        """

        # get current axes
        if axes is None:
            axes = plt.gca()

        # limit of x and classes in (L, H)
        x_lim = axes.get_xlim()
        y_lim = axes.get_ylim()

        # create grid to evaluate model
        # from L to H in 30 parts
        x = np.linspace(x_lim[0], x_lim[1], 30)
        classes = np.linspace(y_lim[0], y_lim[1], 30)

        # create a rectangular grid out of two given one-dimensional arrays
        Y, points = np.meshgrid(classes, x)

        # Stack arrays in sequence vertically
        xy = np.vstack([points.ravel(), Y.ravel()]).T

        P = model.decision_function(xy).reshape(points.shape)

        # plot decision boundary and margins
        axes.contour(points, Y, P, colors='black', levels=[-1, 0, 1], alpha=0.6, linestyles=['--', '-', '--'])

        # plot support vectors
        if plot_support:
            axes.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1,
                         facecolors='none')

    def plot_svm1(self, ax, N: int, random_state: int, cluster_std: float, kernel: str = "rbf", degree: int = None,
                  gamma: str = None) -> None:
        """
            Method to fill the plot,

            Parameters:
                ax: The ax
                N (int): Number of samples
                random_state (int): The random state
                cluster_std (float): The Cluster
                kernel (str): The type of kernel to use
                degree (int): The degree of kernel
                gamma (str): The gamma of kernel
        """

        # create points and classes
        points, classes = make_blobs(n_samples=N, cluster_std=cluster_std, random_state=random_state,
                                     centers=self.NUMBER_OF_PARTITIONS)

        # the dictionary type to keep parameters of any kernel
        fine_dict = dict(kernel=kernel)

        if kernel == "sigmoid":
            if gamma:  fine_dict['gamma'] = gamma

        elif kernel == "rbf":
            if gamma:  fine_dict['gamma'] = gamma

        elif kernel == "poly":
            if degree: fine_dict['degree'] = degree
            if gamma:  fine_dict['gamma'] = gamma

        # make and fit the model
        model = SVC(**fine_dict)
        model.fit(points, classes)

        if ax is None:
            ax = plt.gca()

        # scatter the ax
        ax.scatter(points[:, 0], points[:, 1], c=classes, s=40, cmap='autumn')

        # plot the decision function for a 2D SVC
        self.plot_svc_decision_function(model, ax)

    def plot_svm2(self, ax, N: int, factor: float, noise: float, random_state: int, shuffle: bool = False,
                  kernel: str = "rbf", degree: int = None, gamma: str = None):
        """
            Method to fill the plot,

            Parameters:
                ax: The ax
                N (int): Number of samples
                factor (float): The factor number
                noise (float): The noise number
                random_state (int): The random state
                shuffle (bool): To shuffle the samples or not
                kernel (str): The type of kernel to use
                degree (int): The degree of kernel
                gamma (str): The gamma of kernel
        """

        # create points and classes
        points, classes = make_circles(N, factor=factor, noise=noise, shuffle=shuffle, random_state=random_state)

        # the dictionary type to keep parameters of any kernel
        fine_dict = dict(kernel=kernel)

        if kernel == "sigmoid":
            if gamma:  fine_dict['gamma'] = gamma

        elif kernel == "rbf":
            if gamma:  fine_dict['gamma'] = gamma

        elif kernel == "poly":
            if degree: fine_dict['degree'] = degree
            if gamma:  fine_dict['gamma'] = gamma

        # make and fit the model
        model = SVC(**fine_dict)
        model.fit(points, classes)

        if ax is None:
            ax = plt.gca()

        # scatter the ax
        ax.scatter(points[:, 0], points[:, 1], c=classes, s=40, cmap='autumn')

        # plot the decision function for a 2D SVC
        self.plot_svc_decision_function(model, ax, plot_support=False)


if __name__ == "__main__":
    svm_ = SVM()
    svm_.run_tests_blobs()
    svm_.run_tests_circle()
