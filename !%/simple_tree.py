import os
import argparse


def print_tree(start_path, prefix="", depth=None, current_depth=0):
    if depth and current_depth >= depth:
        return

    try:
        items = os.listdir(start_path)
    except PermissionError:
        return

    for i, item in enumerate(sorted(items)):
        if item.startswith('.'):
            continue

        path = os.path.join(start_path, item)
        is_last = i == len(items) - 1

        print(prefix + ("└── " if is_last else "├── ") + item)

        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension, depth, current_depth + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple directory tree')
    parser.add_argument('path', nargs='?', default='.', help='Directory path')
    parser.add_argument('-d', '--depth', type=int, help='Max depth')
    args = parser.parse_args()

    print(os.path.basename(os.path.abspath(args.path)) + "/")
    print_tree(args.path, depth=args.depth)