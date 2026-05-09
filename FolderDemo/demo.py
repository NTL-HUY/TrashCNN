import json
from pprint import pprint

from TrashDetect.config import Config
import os

def inspect_json_structure(data, name="JSON"):
    print(f"\n{'=' * 50}")
    print(f"STRUCTURE OF {name}")
    print(f"{'=' * 50}")

    # Key cấp cao nhất
    print("\nTop-level keys:")
    for key in data.keys():
        value = data[key]

        print(f"\n[{key}]")
        print(f"Type: {type(value).__name__}")

        # Nếu là list
        if isinstance(value, list):
            print(f"Length: {len(value)}")

            if len(value) > 0:
                first_item = value[0]

                print(f"First item type: {type(first_item).__name__}")

                # Nếu phần tử là dict
                if isinstance(first_item, dict):
                    print("Fields in first item:")

                    for k, v in first_item.items():
                        print(f"  - {k}: {type(v).__name__}")

                    print("\nExample first item:")
                    pprint(first_item)

                else:
                    print("First item:")
                    pprint(first_item)

        # Nếu là dict
        elif isinstance(value, dict):
            print("Inner keys:")
            for k, v in value.items():
                print(f"  - {k}: {type(v).__name__}")
            print("\nExample:")
            pprint(value)
        else:
            print("Value:")
            pprint(value)
    print(f"\n{'=' * 50}\n")

def stats_categories(data: dict):
    categories = data.get("categories", [])
    print("\n{:<10} {:<20} {:<20}".format("ID", "NAME", "SUPERCATEGORY"))
    print("-" * 60)

    for c in categories:
        print("{:<10} {:<20} {:<20}".format(
            c.get("id"),
            c.get("name"),
            c.get("supercategory")
        ))

def main():
    split = 'train'
    with open(os.path.join(Config.DATA_TACO_PATH,split, "_annotations.processed.coco.json")) as f:
        taco = json.load(f)

    # with open(os.path.join(Config.DATA_GLASS_PATH, split ,"_annotations.processed.coco.json")) as f:
    #     glass = json.load(f)
    pprint(taco.keys())
    inspect_json_structure(taco, "TACO")

    print("\n" + "=" * 50)
    print("CATEGORIES")
    print("=" * 50)

    categories = taco.get("categories", [])
    print("\n{:<10} {:<20} {:<20}".format("ID", "NAME", "SUPERCATEGORY"))
    print("-" * 60)

    for c in categories:
        print("{:<10} {:<20} {:<20}".format(
            c.get("id"),
            c.get("name"),
            c.get("supercategory")
        ))

    # pprint(glass)
if __name__ == "__main__":
    main()
