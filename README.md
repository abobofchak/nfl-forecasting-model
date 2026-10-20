# NFL Game Forecasting Model

A Python pipeline that forecasts NFL game margins, win probabilities and total points from public play-by-play data, evaluates itself with a leakage-safe walk-forward backtest, and exports its results as JSON for a companion mobile app ([Film Room](https://github.com/abobofchak/film-room-app)).

> **Educational project.** It is a statistics and software exercise, not betting advice, and it does not place or recommend bets.

## What it does

- **Ratings:** Elo plus EPA per play (garbage-time filtered, split into pass and rush), multi-window form, and shrunken quarterback ratings.
- **Context features:** rest and bye weeks, short weeks, divisional games, surface, roof, wind and cold, travel distance and time zones.
- **Models:** Ridge regression for the margin, logistic regression for the win probability, and a separate model for total points. The National Weather Service forecast feeds only the totals model.
- **Honest evaluation:** walk-forward backtest from 2014. For each season the model trains only on earlier games. Results are compared with the closing betting line, which is used for evaluation only and is not a model input by default (`USE_MARKET_FEATURES = False`).
- **Feature-group ablation:** re-runs the backtest without each feature group to show what actually helps.
- **App export:** one JSON card per game (injuries, weather, quarterbacks, team stats, a points-by-feature explanation of the predicted margin), plus power rankings, standings, a 10,000-run season simulation for playoff odds, game recaps, and a prediction log that grades the model after each week.
- **Automation:** runs on a schedule with GitHub Actions and publishes the JSON to GitHub Pages (the private operations repo; this repo holds the code only).

## Results

| Metric (walk-forward, 2014 to present) | Model | Closing line |
|---|---|---|
| Spread MAE (points) | 10.22 | 9.89 |
| Total points MAE | 10.73 | 10.44 |
| Win log loss | 0.6317 | 0.6107 |
| Games evaluated | 2,929 | 2,929 |

The model trails the closing line by roughly 0.3 points of MAE on spreads and totals. That is expected, since the line is not a model input. The project's value is the leakage-safe method and the honest comparison.

## Reproduce it

```
git clone https://github.com/abobofchak/nfl-forecasting-model.git
cd nfl-forecasting-model
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python nfl_model.py
```

The first run downloads play-by-play for every season since 2011 and caches it in `cache/`, so it is slow once. Useful switches:

- `NFL_ABLATION=0` skips the feature-group test.
- `NFL_SEASON=2025` forces a season.
- `NWS_USER_AGENT="YourApp/1.0 (you@example.com)"` sets the contact string the National Weather Service asks for.
- `python nfl_model.py --college` and `--skills` rebuild only the app's mini-game files.

## Design notes

- **No leakage.** Every feature for a game uses only information available before kickoff. The app's quiz game follows the same rule and was checked against an independent calculation.
- **One file on purpose.** `nfl_model.py` is a single script so it runs anywhere with one command. Splitting it into modules is a planned cleanup.
- **Outputs are separate from code.** The `cards/` folder and the prediction log are git-ignored here.

## Data and credits

- Game, play-by-play, injury and player data: [nflverse](https://nflverse.com), downloaded at run time through `nflreadpy`. The data is not stored or redistributed in this repository. See nflverse's documentation for the data license terms.
- Weather forecasts: [National Weather Service API](https://www.weather.gov/documentation/services-web-api).
- Independent project. Not affiliated with or endorsed by the NFL or any team.

## License

Code is released under the MIT License (see `LICENSE`).

## Author

Alex Bobofchak, Information Science, University of Maryland.
