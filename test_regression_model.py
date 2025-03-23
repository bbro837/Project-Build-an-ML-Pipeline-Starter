if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--param1", type=str, required=True)
    args = parser.parse_args()

    # Implement the regression model testing logic here
    print(f"Testing regression model with param1: {args.param1}")