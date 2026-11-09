
import argparse
import json

from . import data_loader as dl


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--source", required=True,
                    help="where the data came from (stored for provenance)")
    ap.add_argument("--utc-offset", type=float, default=0,
                    help="hours to subtract from naive timestamps to get UTC (3 = Nairobi local time)")
    ap.add_argument(
        "--segments", help="optional CSV: name, latitude, longitude, length_km, free_flow_speed_kmh, speed_limit_kmh, geometry_json")
    ap.add_argument("--db")
    a = ap.parse_args(argv)
    db = dl.Database.from_env(a.db)
    print(json.dumps(dl.import_traffic_csv(
        db, a.csv, a.source, a.utc_offset, a.segments), indent=2))


if __name__ == "__main__":
    main()
