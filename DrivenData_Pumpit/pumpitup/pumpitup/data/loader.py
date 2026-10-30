import duckdb
import pandas as pd
from pathlib import Path


class PumpDataLoader:
    """
    Unified DuckDB-based loader for Pump-It-Up dataset.
    Loads training values, training labels, and test values.
    Provides pandas DataFrames for model pipelines.
    """

    def __init__(self):
        self.con = duckdb.connect(database=":memory:")

    def load_raw_csvs(self, train_values, train_labels, test_values):
        train_values = Path(train_values)
        train_labels = Path(train_labels)
        test_values = Path(test_values)

        # Load CSVs into DuckDB
        self.con.execute(f"""
            CREATE TABLE train_values AS
            SELECT * FROM read_csv_auto('{train_values}');
        """)

        self.con.execute(f"""
            CREATE TABLE train_labels AS
            SELECT * FROM read_csv_auto('{train_labels}');
        """)

        self.con.execute(f"""
            CREATE TABLE test_values AS
            SELECT * FROM read_csv_auto('{test_values}');
        """)

        # Merge training values + labels
        self.con.execute("""
            CREATE TABLE train_full AS
            SELECT tv.*, tl.status_group
            FROM train_values tv
            JOIN train_labels tl
            USING (id);
        """)

    def get_training_dataframe(self):
        return self.con.execute("SELECT * FROM train_full").df()

    def get_test_dataframe(self):
        return self.con.execute("SELECT * FROM test_values").df()

    def close(self):
        self.con.close()