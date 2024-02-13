import sqlite3
import argparse


def convert(a):
    return "".join(chr(int(x, 2)) for x in a.split())


def convert_db(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur2 = con.cursor()

    cur.execute("CREATE TABLE IF NOT EXISTS wiki2(word TEXT PRIMARY KEY, p_e_m BLOB, lower TEXT, freq INTEGER)")
    cur.execute("SELECT word, p_e_m, lower, freq FROM wiki")

    cur2.execute("BEGIN TRANSACTION")
    for w, blob, lower, freq in cur:
        data = convert(blob).encode()
        cur2.execute("INSERT INTO wiki2 VALUES(?, ?, ?, ?)", [w, data, lower, freq])
    cur2.execute("COMMIT")
    cur2.close()

    cur.execute("BEGIN TRANSACTION")
    cur.execute("DROP INDEX idx_p_e_m")
    cur.execute("DROP INDEX idx_lower")
    cur.execute("DROP INDEX idx_freq")
    cur.execute("DROP TABLE wiki")
    cur.execute("ALTER TABLE wiki2 RENAME TO wiki")
    cur.execute("CREATE INDEX idx_p_e_m ON wiki(p_e_m)")
    cur.execute("CREATE INDEX idx_lower ON wiki(lower)")
    cur.execute("CREATE INDEX idx_freq ON wiki(freq)")
    cur.execute("COMMIT")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'db',
        help='Path to database to convert'
    )
    convert_db(vars(parser.parse_args())['db'])
