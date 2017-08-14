The interleaf regressor is based on the idea of achieving higher resolution from a low-resolution system by oversampling.

A practical use case is predicting a continuous variable (regression) that is hard to "pin down" as a function of the input, and yet the classification problem of "will the output be high or low" shows decent performance.

By interleaving and averaging the predictions of many classifiers, each with slightly offset class thresholds, the resolution of the regression output improves.

Define the resolution of the regression (in units of output) as R = (upper bound - lower bound)/(number of classifiers * classifier resolution).

If the target variable is uniformly distributed over its range, the RMSE of an interleaved regression with perfect predictions is R/(2*sqrt(3))

Therefore, for a relatively modest investment of memory (say 64 classifiers), and in a difficult value function space (so that the classifiers can individually resolve only 2 separate classes reliably), interleaved regression gives an RMSE of under 0.23% of the output range if the classification predictions are good.
