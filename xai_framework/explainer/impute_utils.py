import logging


def check_impute_strategy(strategy: str) -> None:
    valid_strategies = {"median", "most_frequent", "zero", "mean", "none"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid impute_strategy '{strategy}'. "
            f"Supported strategies: {valid_strategies}."
        )


def get_impute_fn(strategy: str):
    strategies = {
        "median": lambda col: col.median(),
        "most_frequent": lambda col: col.mode().iloc[0],
        "zero": lambda col: 0,
        "mean": lambda col: col.mean(),
        "none": lambda col: col,
    }
    fn = strategies.get(strategy)
    if fn is None:
        logging.warning(
            "Unknown impute_strategy '%s'. Falling back to median.", strategy
        )
        fn = strategies["median"]
    return fn
