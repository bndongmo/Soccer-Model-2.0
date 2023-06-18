# Soccer-Model-2.0
Improved...in theory, goal is to create a library/module containing functions to facilitate modeling any count data

Soccer Model for Predicting Teams' Variation in Form
This repository contains a soccer model that aims to predict teams' variation in form for soccer games. The motivation behind the model is to identify situations where oddsmakers are slow to react to teams' performance changes, thereby providing potential opportunities for improved predictions.

Dataset
The model is trained on a dataset consisting of approximately 40,000 soccer games. For each game and each team, the following three numbers were collected:

Goals Scored: The number of goals scored by the team.
Oddsmakers' Predicted Goals: The number of goals the oddsmakers predicted the team to score before the game.
Expected Goals: A metric that analyzes in-game performance, specifically the quality of chances and the expected number of goals from those chances.
These numbers are stored in lists for each team. To ensure sufficient data for accurate modeling, teams must have played at least 8 matches and scored at least 1 goal.

Features
To capture teams' variation in form, the following features are computed for each team:

Total Goals Scored / Total Oddsmakers' Predicted Goals: The ratio of the total goals scored by a team to their total pre-match predicted goals by oddsmakers.
Total Goals Scored / Total Expected Goals: The ratio of the total goals scored by a team to their total expected goals based on the quality of chances in-game.
Total Expected Goals / Total Oddsmakers' Predicted Goals: The ratio of the total expected goals based on the quality of chances in-game to the total pre-match predicted goals by oddsmakers.
Similarly, metrics are calculated for the opponents' goals conceded. Additionally, a weighted average ratio is applied to these metrics to account for the temporal aspect, where more recent games carry more weight. The weights decay by a factor of 0.93 for every game that passes. The logarithm of these metrics is taken to define the features.

Furthermore, the null prediction made by sportsbooks is included as a feature, with the logarithm applied.

Modeling Approach
The model is based on generalized Poisson regression, with the aim of predicting two parameters: lambda and alpha. Lambda represents the predicted count, while alpha determines the dispersion of the distribution. A tanh activation function is used for alpha since it can take on negative values.

To optimize cross-validation performance, various methods in scipy.minimize are experimented with. Custom callbacks are implemented to enable early stopping when cross-validation performance decreases. Further regularization is applied using elastic net, and Bayesian optimization is used to determine the best l1/l2 weights for lambda and alpha separately. Early stopping is employed to optimize the number of iterations.

The final model is trained using the cg method for 40 iterations, with moderate regularization.

Coefficients
The following are the coefficients obtained for lambda and alpha:

Sorted coefficients for lambda:

Parameter             Coef
-----------------  -------
loglines           1.0953

wxgoals_lines_for  0.2040

wgoals_xgoals_for -0.1385

wgoals_lines_ag   -0.0973

goals_xgoals_for   0.0939

xgoals_lines_for  -0.0810

goals_lines_for   -0.0616

xgoals_lines_ag    0.0580

wxgoals_lines_ag  -0.0476

intercept         -0.0462

wgoals_lines_for   0.0383

goals_lines_ag     0.0244

wgoals_xgoals_ag   0.0050

goals_xgoals_ag    0.0041
Sorted coefficients for alpha:

Parameter             Coef
-----------------  -------
loglines          -0.3108

wgoals_xgoals_ag   0.1128

goals_xgoals_for   0.0965

intercept          0.0933

wgoals_xgoals_for  0.0689

wgoals_lines_ag    0.0573

wgoals_lines_for   0.0455

goals_lines_for    0.0403

goals_xgoals_ag    0.0402

wxgoals_lines_for  0.0333

xgoals_lines_ag    0.0151

xgoals_lines_for   0.0087

goals_lines_ag     0.0023

wxgoals_lines_ag  -0.0006
