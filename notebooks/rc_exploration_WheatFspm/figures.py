import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve


def learning_validation_curves(
    model,
    X,
    y,
    groups,
    folds,
    scorer,
    param_name,
    param_range,
    n_train_sizes=10,
    validation_curve_scale="log",
):

    # Generate data for learning curves
    train_sizes = np.linspace(0.1, 1.0, n_train_sizes)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, groups=groups, train_sizes=train_sizes, cv=folds, scoring=scorer,
    )
    train_scores = -train_scores
    test_scores = -test_scores
    learning_curve_train_mean = np.mean(train_scores, axis=1)
    learning_curve_train_std = np.std(train_scores, axis=1)
    learning_curve_val_mean = np.mean(test_scores, axis=1)
    learning_curve_val_std = np.std(test_scores, axis=1)

    # Generate data for validation curves
    train_scores, test_scores = validation_curve(
        model,
        X,
        y,
        groups=groups,
        param_name=param_name,
        param_range=param_range,
        cv=folds,
        scoring=scorer,
    )
    train_scores = -train_scores
    test_scores = -test_scores
    validation_curve_train_mean = np.mean(train_scores, axis=1)
    validation_curve_train_std = np.std(train_scores, axis=1)
    validation_curve_val_mean = np.mean(test_scores, axis=1)
    validation_curve_val_std = np.std(test_scores, axis=1)

    # Create plot
    with plt.style.context("ggplot"):
        fig, (ax_lc, ax_vc) = plt.subplots(
            1, 2, figsize=(12, 5), constrained_layout=True
        )

        # Learning curve
        ax_lc.set_xticks(train_sizes_abs)
        ax_lc.set_ylim(0, 1.5)
        #   Training curve
        ax_lc.fill_between(
            train_sizes_abs,
            learning_curve_train_mean + learning_curve_train_std,
            learning_curve_train_mean - learning_curve_train_std,
            alpha=0.2,
            color="blue",
        )
        ax_lc.plot(
            train_sizes_abs,
            learning_curve_train_mean,
            color="blue",
            label="Training score",
        )
        ax_lc.scatter(train_sizes_abs, learning_curve_train_mean, color="blue")
        #   Validation curve
        ax_lc.fill_between(
            train_sizes_abs,
            learning_curve_val_mean + learning_curve_val_std,
            learning_curve_val_mean - learning_curve_val_std,
            alpha=0.2,
            color="red",
        )
        ax_lc.plot(
            train_sizes_abs, learning_curve_val_mean, color="red", label="CV score"
        )
        ax_lc.scatter(train_sizes_abs, learning_curve_val_mean, color="red")
        #   Labeling
        ax_lc.legend(loc="upper right")
        ax_lc.axhline(1, dashes=(2, 2), color="black")
        ax_lc.text(train_sizes_abs[0] + 3, 1.035, "baseline", fontsize="large")
        ax_lc.set_xlabel("Training samples")
        ax_lc.set_ylabel("score")
        ax_lc.set_title("Learning curve")

        # Validation curve
        ax_vc.set_xscale(validation_curve_scale)
        ax_vc.set_ylim(0, 1.5)
        ax_vc.set_xlim(min(param_range), max(param_range))
        #   Training curve
        ax_vc.plot(
            param_range,
            validation_curve_train_mean,
            color="blue",
            label="Training acc.",
        )
        ax_vc.fill_between(
            param_range,
            validation_curve_train_mean + validation_curve_train_std,
            validation_curve_train_mean - validation_curve_train_std,
            alpha=0.2,
            color="blue",
        )
        #   Validation curve
        ax_vc.plot(param_range, validation_curve_val_mean, color="red", label="CV acc.")
        ax_vc.fill_between(
            param_range,
            validation_curve_val_mean + validation_curve_val_std,
            validation_curve_val_mean - validation_curve_val_std,
            alpha=0.2,
            color="red",
        )
        #   Labeling
        ax_vc.legend(loc="best")
        ax_vc.axhline(1, dashes=(2, 2), color="black")
        ax_vc.text(0.60, 1.035, "baseline", fontsize="large")
        ax_vc.set_xlabel(param_name)
        ax_vc.set_ylabel("score")
        ax_vc.set_title(f"Validation curve ({param_name})")

        fig.suptitle("Model evaluation", size="xx-large")

        plt.show()

